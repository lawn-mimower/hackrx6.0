import concurrent.futures
import cv2
import fitz  # PyMuPDF
import numpy as np
import re
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pinecone import Pinecone, ServerlessSpec # Import ServerlessSpec
from typing import List, Dict, Any
import logging
import asyncio
from config import Config

logger = logging.getLogger(__name__)

# --- Corrected Initialization ---
try:
    # Initialize Pinecone Client
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    # Initialize models
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL)
except Exception as e:
    logger.fatal(f"CRITICAL: Failed during model or client initialization in embedder.py. Error: {e}")
    exit()


def extract_and_format_tables(page):
    # This function remains the same
    table_chunks = []
    try:
        tables = page.find_tables()
        for i, table in enumerate(tables):
            table_data = table.extract()
            if not table_data:
                continue

            header = " | ".join(map(str, table_data[0]))
            separator = " | ".join(["---"] * len(table_data[0]))
            body = "\n".join([" | ".join(map(str, row)) for row in table_data[1:]])

            markdown_table = f"Table on page {page.number + 1}:\n{header}\n{separator}\n{body}"
            table_chunks.append({
                "text": markdown_table,
                "type": "table",
                "page_number": page.number + 1,
                "bbox": table.bbox
            })
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
    return table_chunks

def split_text_intelligently(text, tokenizer, max_tokens, overlap):
    # This function remains the same
    sentences = re.split(r'(?<=[.?!])\s+', text.replace("\n", " "))
    chunks = []
    current_chunk_tokens = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_tokens) > max_tokens:
            sub_step = max_tokens - overlap
            for i in range(0, len(sentence_tokens), sub_step):
                chunks.append(tokenizer.decode(sentence_tokens[i:i + max_tokens]))
            continue
        if len(current_chunk_tokens) + len(sentence_tokens) <= max_tokens:
            current_chunk_tokens.extend(sentence_tokens)
        else:
            chunks.append(tokenizer.decode(current_chunk_tokens))
            current_chunk_tokens = sentence_tokens[-overlap:] if overlap > 0 else sentence_tokens
    if current_chunk_tokens:
        chunks.append(tokenizer.decode(current_chunk_tokens))
    return chunks

def process_page(page_info):
    # This function remains the same
    page_num, pdf_path, areas_to_ignore = page_info
    logger.info(f"Processing page {page_num}...")
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        page_image_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_image = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = page_image.shape
        pdf_width, pdf_height = page.rect.width, page.rect.height
        x_scale = pdf_width / img_width
        y_scale = pdf_height / img_height
        for bbox in areas_to_ignore:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(page_image, (int(x0 / x_scale), int(y0 / y_scale)), (int(x1 / x_scale), int(y1 / y_scale)), (0, 0, 0), -1)
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_chunks = []
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    for contour in sorted_contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blob_bbox = fitz.Rect(x * x_scale, y * y_scale, (x + w) * x_scale, (y + h) * y_scale)
        chunk_text = ""
        font_sizes, is_bold_list = [], []
        for block in blocks:
            if block['type'] == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if fitz.Rect(span['bbox']).intersects(blob_bbox):
                            chunk_text += span['text'] + " "
                            font_sizes.append(span['size'])
                            is_bold_list.append(span['flags'] & 2**4)
        chunk_text = chunk_text.strip()
        if chunk_text:
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            is_heading = any(is_bold_list) or (avg_font_size > 12)
            page_chunks.append({"text": chunk_text, "page_number": page_num + 1, "type": "heading" if is_heading else "paragraph", "avg_font_size": avg_font_size})
    logger.info(f"Extracted {len(page_chunks)} chunks from page {page_num}")
    return page_chunks

async def store_embeddings(pages_to_process: List[int], pdf_path: str, doc_namespace: str):
    """
    Process specified pages and store embeddings in Pinecone.
    """
    try:
        # --- Corrected Pinecone Interaction ---
        index_name = Config.PINECONE_INDEX_NAME
        
        # 1. Check if index exists using the 'pc' client
        if index_name not in pc.list_indexes().names():
            logger.info(f"Index '{index_name}' not found. Creating a new serverless index...")
            # 2. Create index using the 'pc' client and modern spec
            pc.create_index(
                name=index_name,
                dimension=Config.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws', # Or 'gcp', 'azure'
                    region='us-west-2' # Choose a region
                )
            )
            logger.info(f"Index '{index_name}' created successfully.")
        
        # 3. Connect to the index
        index = pc.Index(index_name)
        
        all_document_chunks = []
        areas_to_ignore_by_page = {i: [] for i in pages_to_process}
        
        with fitz.open(pdf_path) as doc:
            for page_num in pages_to_process:
                if page_num < len(doc):
                    page = doc.load_page(page_num)
                    table_chunks = extract_and_format_tables(page)
                    if table_chunks:
                        all_document_chunks.extend(table_chunks)
                        areas_to_ignore_by_page[page_num] = [chunk['bbox'] for chunk in table_chunks]
        
        pages_info = [(num, pdf_path, areas_to_ignore_by_page.get(num, [])) for num in pages_to_process]
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [loop.run_in_executor(executor, process_page, page_info) for page_info in pages_info]
            results = await asyncio.gather(*futures)
            for page_result in results:
                all_document_chunks.extend(page_result)
        
        logger.info(f"Total raw chunks extracted: {len(all_document_chunks)}")
        
        final_chunks_with_metadata = []
        for chunk_info in all_document_chunks:
            text = chunk_info['text']
            if len(tokenizer.encode(text)) <= Config.MAX_TOKENS_PER_CHUNK:
                final_chunks_with_metadata.append(chunk_info)
            else:
                sub_chunks = split_text_intelligently(text, tokenizer, Config.MAX_TOKENS_PER_CHUNK, Config.CHUNK_OVERLAP)
                for sub_chunk_text in sub_chunks:
                    new_chunk_info = chunk_info.copy()
                    new_chunk_info['text'] = sub_chunk_text
                    final_chunks_with_metadata.append(new_chunk_info)
        
        logger.info(f"Total final chunks: {len(final_chunks_with_metadata)}")
        
        texts_to_embed = [chunk['text'] for chunk in final_chunks_with_metadata]
        embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(final_chunks_with_metadata, embeddings)):
            vector_id = f"{doc_namespace}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {"text": chunk['text'], "page_number": chunk['page_number'], "type": chunk.get('type', 'paragraph'), "namespace": doc_namespace}
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=doc_namespace)
        
        logger.info(f"Successfully stored {len(vectors)} embeddings in Pinecone")
        
    except Exception as e:
        logger.error(f"Embedding storage error: {e}", exc_info=True)
        raise