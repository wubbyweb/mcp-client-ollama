import pytest
import os
import shutil
import tempfile
from server.vector_store import VectorStore

@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def vector_store(temp_db_dir):
    """Create a VectorStore instance with temporary storage."""
    return VectorStore(persist_directory=temp_db_dir)

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return {
        "chunks": [
            "This is the first test chunk.",
            "This is the second test chunk.",
            "This is the third test chunk."
        ],
        "embeddings": [
            [0.1] * 1536,  # Mock embeddings with correct dimension
            [0.2] * 1536,
            [0.3] * 1536
        ],
        "metadata": [
            {"source": "test1.md", "chunk_index": 0, "last_updated": "123456"},
            {"source": "test1.md", "chunk_index": 1, "last_updated": "123456"},
            {"source": "test1.md", "chunk_index": 2, "last_updated": "123456"}
        ]
    }

def test_vector_store_initialization(temp_db_dir):
    """Test VectorStore initialization and collection creation."""
    store = VectorStore(persist_directory=temp_db_dir)
    assert store.client is not None
    assert store.collection is not None
    assert os.path.exists(temp_db_dir)

def test_add_documents(vector_store, sample_documents):
    """Test adding documents to the vector store."""
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Verify first document was added
    doc_id = f"{sample_documents['metadata'][0]['source']}_{sample_documents['metadata'][0]['chunk_index']}"
    result = vector_store.get_document_by_id(doc_id)
    
    assert result["document"] == sample_documents["chunks"][0]
    assert result["metadata"] == sample_documents["metadata"][0]

def test_search_similar(vector_store, sample_documents):
    """Test searching for similar documents."""
    # Add sample documents
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Search using first document's embedding
    results = vector_store.search_similar(
        query_embedding=sample_documents["embeddings"][0],
        n_results=2
    )
    
    assert len(results["documents"]) == 2
    assert len(results["metadatas"]) == 2
    assert len(results["distances"]) == 2

def test_update_document(vector_store, sample_documents):
    """Test updating a document in the vector store."""
    # Add initial documents
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Update first document
    doc_id = f"{sample_documents['metadata'][0]['source']}_{sample_documents['metadata'][0]['chunk_index']}"
    updated_chunk = "This is an updated chunk."
    updated_embedding = [0.5] * 1536
    updated_metadata = {
        "source": "test1.md",
        "chunk_index": 0,
        "last_updated": "123457"
    }
    
    vector_store.update_document(
        document_id=doc_id,
        chunk=updated_chunk,
        embedding=updated_embedding,
        metadata=updated_metadata
    )
    
    # Verify update
    result = vector_store.get_document_by_id(doc_id)
    assert result["document"] == updated_chunk
    assert result["metadata"] == updated_metadata

def test_delete_document(vector_store, sample_documents):
    """Test deleting a document from the vector store."""
    # Add initial documents
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Delete first document
    doc_id = f"{sample_documents['metadata'][0]['source']}_{sample_documents['metadata'][0]['chunk_index']}"
    vector_store.delete_document(doc_id)
    
    # Verify deletion
    with pytest.raises(Exception):
        vector_store.get_document_by_id(doc_id)

def test_clear_collection(vector_store, sample_documents):
    """Test clearing all documents from the collection."""
    # Add initial documents
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Clear collection
    vector_store.clear_collection()
    
    # Verify all documents are removed
    with pytest.raises(Exception):
        doc_id = f"{sample_documents['metadata'][0]['source']}_{sample_documents['metadata'][0]['chunk_index']}"
        vector_store.get_document_by_id(doc_id)

def test_search_with_metadata_filter(vector_store, sample_documents):
    """Test searching with metadata filters."""
    # Add initial documents
    vector_store.add_documents(
        chunks=sample_documents["chunks"],
        embeddings=sample_documents["embeddings"],
        metadata=sample_documents["metadata"]
    )
    
    # Search with metadata filter
    results = vector_store.search_similar(
        query_embedding=sample_documents["embeddings"][0],
        n_results=3,
        metadata_filter={"source": "test1.md"}
    )
    
    assert len(results["documents"]) > 0
    assert all(meta["source"] == "test1.md" for meta in results["metadatas"])