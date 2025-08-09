import os
import hashlib
import httpx
import aiofiles
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    async def get_or_download(self, url: str, url_hash: str) -> Path:
        """
        Check cache for document or download if not present.
        """
        # Generate cache filename
        cache_path = self.cache_dir / f"{url_hash}.pdf"
        
        # Check if file exists in cache
        if cache_path.exists():
            logger.info(f"Using cached document: {cache_path}")
            return cache_path
        
        # Download document
        logger.info(f"Downloading document from: {url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Save to cache
            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(response.content)
                
        logger.info(f"Document cached at: {cache_path}")
        return cache_path
    
    def clear_cache(self):
        """Clear all cached documents."""
        for file in self.cache_dir.glob("*"):
            if file.is_file():
                file.unlink()