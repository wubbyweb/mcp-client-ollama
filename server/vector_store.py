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
        try:
            ids = [f"{meta['source']}_{meta['chunk_index']}" for meta in metadata]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata
            )
            print(f"Successfully added {len(chunks)} documents to the collection.")
        except Exception as e:
            print(f"Error adding documents: {e}")
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
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            print("Cleared all documents from the collection.")
        except Exception as e:
            print(f"Error clearing ChromaDB collection: {e}")
            raise

    def list_collections_and_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """List all collections and embeddings within each collection."""
        try:
            result = {}
            collection_data = self.collection.get()
            result["documents"] = {
                "count": len(collection_data["ids"]),
                "ids": collection_data["ids"],
                "metadatas": collection_data["metadatas"]
            }
            if "embeddings" in collection_data and collection_data["embeddings"] is not None:
                result["documents"]["embeddings"] = collection_data["embeddings"]
            else:
                print("Warning: Embeddings are not available in the collection data.")
            return result
        except Exception as e:
            print(f"Error listing collections and embeddings: {e}")
            raise

    def list_collections(self) -> List[str]:
        """List all collections in the vector store."""
        try:
            return self.client.list_collections()
        except Exception as e:
            print(f"Error listing collections: {e}")
            raise

    def clear_all_embeddings(self) -> None:
        """Clear all embeddings from all collections in the vector store."""
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            print("Cleared all embeddings from the collection.")
        except Exception as e:
            print(f"Error clearing all embeddings: {e}")
            raise