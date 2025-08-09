import google.generativeai as genai
import torch
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone
from typing import List
import logging
import asyncio
import os
from config import Config  # Assumes config.py exists

# --- Initial Setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model and Client Initialization ---

# Configure Gemini
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
except Exception as e:
    logger.fatal(f"CRITICAL: Failed to configure Gemini API. Error: {e}")
    exit()

# Load the SentenceTransformer model with robust error handling
try:
    logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.fatal(f"CRITICAL: Failed to load embedding model. Cannot continue. Error: {e}")
    exit() # Stop the application if the core model can't load

# Initialize Pinecone client and define the index object once
try:
    logger.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    logger.info(f"Successfully connected to Pinecone index '{Config.PINECONE_INDEX_NAME}'.")
except Exception as e:
    logger.fatal(f"CRITICAL: Failed to initialize Pinecone. Error: {e}")
    exit()

# --- Core Functions ---
async def retrieve_and_answer(elaborated_questions: List[str], doc_namespace: str) -> List[str]:
    """
    Retrieve relevant chunks and generate answers for a batch of questions
    using single, optimized calls to Pinecone and Gemini.
    """
    try:
        if not elaborated_questions:
            return []

        logger.info(f"Starting batch retrieval for {len(elaborated_questions)} questions...")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # 1. BATCH EMBEDDING: Generate embeddings for all questions at once.
        query_embeddings = embedding_model.encode(elaborated_questions, convert_to_tensor=True)
        query_vectors = [emb.cpu().numpy().tolist() for emb in query_embeddings]

        # 2. BATCH PINECONE QUERY: Send multiple queries in a single network call.
        # Note: The Pinecone SDK's query method handles one vector at a time.
        # We use asyncio.gather to run them concurrently, which is much faster than a sequential loop.
        async def query_pinecone(vector):
            return index.query(
                vector=vector,
                top_k=3,
                include_metadata=True,
                namespace=doc_namespace
            )

        # Run all Pinecone queries concurrently
        query_responses = await asyncio.gather(*[query_pinecone(v) for v in query_vectors])

        # 3. BATCH PROMPT PREPARATION: Prepare all contexts and prompts.
        prompts = []
        for i, response in enumerate(query_responses):
            context_chunks = [
                match['metadata']['text']
                for match in response.get('matches', [])
                if match.get('score', 0) > 0.5
            ]

            if not context_chunks:
                # If no context, create a prompt that allows the model to say so.
                context = "No relevant context found in the document."
            else:
                context = "\n---\n".join(context_chunks)

            prompt = f"""Based ONLY on the context provided below, answer the following question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question:
{elaborated_questions[i]}

Answer:"""
            prompts.append(prompt)

        # 4. BATCH ANSWER GENERATION: Send all prompts to Gemini concurrently.
        logger.info(f"Generating answers for {len(prompts)} prompts in a batch...")
        async def generate(prompt):
            try:
                return await model.generate_content_async(prompt)
            except Exception as e:
                logger.error(f"Error generating content for a prompt: {e}")
                return None # Return None on failure for a specific prompt

        # Run all Gemini generations concurrently
        gemini_responses = await asyncio.gather(*[generate(p) for p in prompts])
        
        # Process responses, handling potential errors for individual generations
        final_answers = []
        for res in gemini_responses:
            if res:
                final_answers.append(res.text.strip())
            else:
                final_answers.append("Sorry, an error occurred while generating the answer for this question.")

        logger.info("Successfully generated all answers.")
        return final_answers

    except Exception as e:
        logger.error(f"A major error occurred in the batch retrieval process: {e}")
        return [f"A major retrieval error occurred: {e}"] * len(elaborated_questions)

async def retrieve_with_reranking(
    question: str,
    doc_namespace: str,
    top_k: int = 10,  # Retrieve more candidates for reranking
    rerank_top: int = 3
) -> List[str]:
    """
    Advanced retrieval with bi-encoder reranking for better accuracy.
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(question, convert_to_tensor=True)

        # 1. Retrieve a larger set of initial candidates
        query_response = index.query(
            vector=query_embedding.cpu().numpy().tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=doc_namespace
        )

        matches = query_response.get('matches', [])
        if not matches:
            return []

        # 2. Rerank using a more precise similarity calculation
        chunk_texts = [match['metadata']['text'] for match in matches]
        
        # Calculate cosine similarity between the question and all chunk texts
        similarities = util.cos_sim(query_embedding, embedding_model.encode(chunk_texts, convert_to_tensor=True))[0]

        # Combine scores and sort
        reranked_chunks = [
            (match['metadata']['text'], similarities[i].item())
            for i, match in enumerate(matches)
        ]
        
        reranked_chunks.sort(key=lambda x: x[1], reverse=True)

        # 3. Return the top results after reranking
        return [text for text, score in reranked_chunks[:rerank_top]]

    except Exception as e:
        logger.error(f"Reranking error: {e}")
        return []

# --- Example Usage Block ---

async def main():
    """Main function to demonstrate and test the retrieval functions."""
    print("--- Running Demo ---")
    
    # Ensure you have a namespace with data in your Pinecone index
    TEST_NAMESPACE = "your-test-namespace"
    
    # --- Demo 1: retrieve_and_answer ---
    test_questions = ["What is the main topic of the document?", "Summarize the key findings."]
    print(f"\n1. Testing retrieve_and_answer with {len(test_questions)} questions...")
    answers = await retrieve_and_answer(test_questions, doc_namespace=TEST_NAMESPACE)
    for q, a in zip(test_questions, answers):
        print(f"\nQ: {q}\nA: {a}")

    # --- Demo 2: retrieve_with_reranking ---
    test_rerank_question = "What are the limitations of the proposed method?"
    print(f"\n\n2. Testing retrieve_with_reranking for the question: '{test_rerank_question}'")
    reranked_context = await retrieve_with_reranking(test_rerank_question, doc_namespace=TEST_NAMESPACE)
    if reranked_context:
        print("Top 3 reranked contexts found:")
        for i, context in enumerate(reranked_context):
            print(f"  {i+1}. {context[:150]}...") # Print snippet
    else:
        print("No context found after reranking.")


if __name__ == "__main__":
    # This block allows you to run the script directly for testing
    # Example: python your_script_name.py
    asyncio.run(main())