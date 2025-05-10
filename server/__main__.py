import uvicorn
from pathlib import Path
from .config import Settings
from .api import app

def main():
    """Main entry point for the MCP RAG server."""
    # Load settings
    settings = Settings()
    
    # Ensure required directories exist
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_db_dir).mkdir(parents=True, exist_ok=True)
    
    # Run server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()