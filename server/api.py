from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
from pydantic import BaseModel

from .config import Settings
from .document_processor import DocumentProcessor
from .vector_store import VectorStore

# Request/Response Models
class ProcessDirectoryRequest(BaseModel):
    directory: str

class GenerateContextRequest(BaseModel):
    query: str
    n_results: Optional[int] = 3

class Context(BaseModel):
    content: str
    metadata: Dict[str, Any]
    relevance_score: float

class GenerateContextResponse(BaseModel):
    query: str
    contexts: List[Context]

# Initialize FastAPI app
app = FastAPI(title="MCP RAG Server")

# Load settings
settings = Settings()

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor(settings.openai_api_key)
vector_store = VectorStore(settings.chroma_db_dir)

async def get_document_processor():
    """Dependency for document processor."""
    return document_processor

async def get_vector_store():
    """Dependency for vector store."""
    return vector_store

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/documents/process")
async def process_documents(
    request: ProcessDirectoryRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    store: VectorStore = Depends(get_vector_store)
):
    """Process all markdown files in a directory."""
    try:
        # Ensure directory exists
        if not Path(request.directory).exists():
            raise HTTPException(status_code=404, detail="Directory not found")
            
        # Process documents
        results = await doc_processor.process_directory(request.directory)
        
        # Store in vector store
        for result in results:
            store.add_documents(
                chunks=result["chunks"],
                embeddings=result["embeddings"],
                metadata=result["metadata"]
            )
            
        return {
            "status": "success",
            "processed_files": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/generate", response_model=GenerateContextResponse)
async def generate_context(
    request: GenerateContextRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    store: VectorStore = Depends(get_vector_store)
):
    """Generate context for a query using RAG."""
    try:
        # Generate embedding for query
        query_embedding = await doc_processor.generate_embeddings([request.query])
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
            
        # Search for similar contexts
        results = store.search_similar(
            query_embedding=query_embedding[0],
            n_results=request.n_results
        )
        
        # Format response
        contexts = []
        for i, (doc, meta, distance) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            contexts.append(Context(
                content=doc,
                metadata=meta,
                relevance_score=1 - distance  # Convert distance to similarity score
            ))
            
        return GenerateContextResponse(
            query=request.query,
            contexts=contexts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/list")
async def list_documents(
    store: VectorStore = Depends(get_vector_store)
):
    """List all processed documents."""
    try:
        # Get all documents from the collection
        results = store.collection.get()
        
        # Group by source file
        documents = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            source = meta["source"]
            if source not in documents:
                documents[source] = {
                    "chunks": [],
                    "last_updated": meta["last_updated"]
                }
            documents[source]["chunks"].append({
                "content": doc,
                "chunk_index": meta["chunk_index"]
            })
            
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{source}")
async def delete_document(
    source: str,
    store: VectorStore = Depends(get_vector_store)
):
    """Delete all chunks from a specific source document."""
    try:
        # Get all documents
        results = store.collection.get()
        
        # Find and delete chunks from the source
        deleted = 0
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if meta["source"] == source:
                store.delete_document(doc_id)
                deleted += 1
                
        return {
            "status": "success",
            "deleted_chunks": deleted
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)