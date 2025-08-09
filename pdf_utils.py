import fitz  # PyMuPDF
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import logging
import asyncio
from config import Config
import re
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)

async def elaborate_questions(questions: List[str]) -> List[str]:
    """
    Use Gemini to elaborate a batch of questions in a single API call.
    """
    if not questions:
        return []

    try:
        # Use a valid, current model name
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # 1. Create a single prompt for batch processing
        # The prompt instructs the LLM to maintain order and use a specific format.
        numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        prompt = f"""
        You are an expert at query expansion. Elaborate each of the following questions to 
        include relevant keywords, synonyms, and context for a document search engine.

        **Instructions:**
        1.  Process all questions provided below.
        2.  Return the elaborated version for EACH question.
        3.  **Crucially, maintain the original order.** Respond with a numbered list where each 
            number corresponds to the original question number. Do not add any preamble or
            concluding text, only the numbered list of elaborated questions.

        **Original Questions:**
        {numbered_questions}

        **Elaborated Questions:**
        """

        # 2. Make a single API call for the entire batch
        logger.info(f"Elaborating a batch of {len(questions)} questions...")
        response = await model.generate_content_async(prompt)
        
        # 3. Parse the structured output
        # The regex splits the response by lines that start with a number and a dot.
        # e.g., "1. ...", "2. ...", etc.
        elaborated_text = response.text.strip()
        
        # Split the string into a list of elaborated questions, removing the numbering
        # This regex handles potential variations in spacing after the number.
        elaborated = [re.sub(r'^\d+\.\s*', '', line).strip() for line in elaborated_text.split('\n') if re.match(r'^\d+\.\s*', line)]

        # 4. Validate the output
        if len(elaborated) == len(questions):
            logger.info("Successfully elaborated questions in a batch.")
            return elaborated
        else:
            logger.warning("Batch elaboration failed to return the correct number of questions. Falling back to original questions.")
            return questions

    except Exception as e:
        logger.error(f"Batch question elaboration error: {e}")
        # Fallback to original questions if the batch process fails
        return questions


async def page_finder(elaborated_questions: List[str], pdf_path: str) -> List[int]:
    """
    Find relevant pages using TF-IDF keyword matching.
    Returns 0-indexed page numbers.
    """
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Extract text from all pages
        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            page_texts.append(text)
        
        doc.close()
        
        # Combine all questions into one query
        combined_query = " ".join(elaborated_questions)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features for efficiency
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Fit and transform page texts
        try:
            page_vectors = vectorizer.fit_transform(page_texts)
            query_vector = vectorizer.transform([combined_query])
        except ValueError:
            # If no features found, return all pages (fallback)
            logger.warning("No TF-IDF features found, returning all pages")
            return list(range(min(len(page_texts), 25)))  # Limit to 25 pages max
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, page_vectors).flatten()
        
        # Get top pages (threshold-based selection)
        threshold = 0.1  # Adjust based on testing
        relevant_pages = np.where(similarities > threshold)[0]
        
        # If no pages meet threshold, get top 10
        if len(relevant_pages) == 0:
            relevant_pages = np.argsort(similarities)[-10:][::-1]
        
        # Sort by page number and limit to 25 pages for efficiency
        relevant_pages = sorted(relevant_pages.tolist())[:25]
        
        logger.info(f"Found {len(relevant_pages)} relevant pages using TF-IDF")
        return relevant_pages
        
    except Exception as e:
        logger.error(f"Page finding error: {str(e)}")
        # Return first 10 pages as fallback
        return list(range(min(10, len(doc))))

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text using TF-IDF.
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        vectorizer.fit([text])
        feature_names = vectorizer.get_feature_names_out()
        
        return feature_names.tolist()
        
    except:
        return []