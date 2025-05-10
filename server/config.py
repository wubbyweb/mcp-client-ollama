from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Server configuration settings."""
    
    # OpenAI API configuration
    openai_api_key: str
    
    # Document directory configuration
    documents_dir: str = "./documents"
    
    # ChromaDB configuration
    chroma_db_dir: str = "./chroma_db"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS configuration
    allowed_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"