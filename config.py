import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_HOST = os.getenv("PINECONE_HOST")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BEARER_TOKEN = os.getenv("BEARER_TOKEN")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Processing settings
    MAX_TOKENS_PER_CHUNK = 256
    CHUNK_OVERLAP = 30
    EMBEDDING_MODEL = './models/sentence-transformers/all-MiniLM-L6-v2'
    EMBEDDING_DIMENSION = 384
    
    # Cache settings
    CACHE_DIR = "./cache"
    
    # Timeout settings
    REQUEST_TIMEOUT = 30
    
    # Document formats
    SUPPORTED_FORMATS = {
        'pdf': ['application/pdf'],
        'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        'doc': ['application/msword'],
        'eml': ['message/rfc822', 'text/plain'],
        'msg': ['application/vnd.ms-outlook'],
        'html': ['text/html'],
        'htm': ['text/html'],
        'txt': ['text/plain']
    }