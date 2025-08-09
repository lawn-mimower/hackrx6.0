from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import asyncio
import hashlib
import logging
from pathlib import Path

from config import Config
from cache_manager import CacheManager
from document_converter import DocumentConverter
from pdf_utils import elaborate_questions, page_finder
from embedder import store_embeddings
from retrieval import retrieve_and_answer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 Retrieval System",
    description="LLM-powered document query system",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
class QueryResponse(BaseModel):
    answers: List[str]

# Initialize managers
cache_manager = CacheManager(Config.CACHE_DIR)
doc_converter = DocumentConverter()

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != Config.BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Retrieval System API", "status": "online"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process document queries using LLM-powered retrieval.
    """
    try:
        # Set timeout
        async def process_with_timeout():
            return await asyncio.wait_for(
                process_document_queries(request),
                timeout=Config.REQUEST_TIMEOUT
            )
        
        result = await process_with_timeout()
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Request processing exceeded 30 seconds"
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

async def process_document_queries(request: QueryRequest) -> QueryResponse:
    """
    Main processing pipeline.
    """
    try:
        # 1. Get document URL hash for namespacing
        url_hash = hashlib.md5(str(request.documents).encode()).hexdigest()
        
        # 2. Check cache or download document
        logger.info(f"Processing document: {request.documents}")
        cached_pdf_path = await cache_manager.get_or_download(
            str(request.documents), 
            url_hash
        )
        
        # 3. Convert to PDF if needed
        pdf_path = await doc_converter.ensure_pdf(cached_pdf_path)
        
        # 4. Elaborate questions using Gemini
        logger.info("Elaborating questions...")
        elaborated_questions = await elaborate_questions(request.questions)
        
        # 5. Find relevant pages using TF-IDF
        logger.info("Finding relevant pages...")
        pages_to_process = await page_finder(elaborated_questions, pdf_path)
        logger.info(f"Found {len(pages_to_process)} relevant pages")
        
        # 6. Store embeddings in Pinecone
        logger.info("Generating and storing embeddings...")
        await store_embeddings(pages_to_process, pdf_path, url_hash)
        
        # 7. Retrieve and generate answers
        logger.info("Generating answers...")
        answers = await retrieve_and_answer(elaborated_questions, url_hash)
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)