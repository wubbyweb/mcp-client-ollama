from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB with persistence."""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self, name: str = "documents"):
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name)
        except ValueError:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

    def add_documents(self, 
                     chunks: List[str],
                     embeddings: List[List[float]],
                     metadata: List[Dict[str, Any]]) -> None:
        """Add document chunks and their embeddings to the vector store."""
        # Generate unique IDs for each chunk
        ids = [f"{meta['source']}_{meta['chunk_index']}" for meta in metadata]
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise

    def search_similar(self, 
                      query_embedding: List[float],
                      n_results: int = 3,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for similar documents using query embedding."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter
            )
            
            return {
                "documents": results["documents"][0],  # First query's results
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0]
            }
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            raise

    def update_document(self,
                       document_id: str,
                       chunk: str,
                       embedding: List[float],
                       metadata: Dict[str, Any]) -> None:
        """Update an existing document in the vector store."""
        try:
            self.collection.update(
                ids=[document_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"Error updating document in ChromaDB: {e}")
            raise

    def delete_document(self, document_id: str) -> None:
        """Delete a document from the vector store."""
        try:
            self.collection.delete(ids=[document_id])
        except Exception as e:
            print(f"Error deleting document from ChromaDB: {e}")
            raise

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """Retrieve a specific document by ID."""
        try:
            result = self.collection.get(ids=[document_id])
            return {
                "document": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        except Exception as e:
            print(f"Error retrieving document from ChromaDB: {e}")
            raise

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.collection.delete()
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"Error clearing ChromaDB collection: {e}")
            raise