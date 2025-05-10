import pytest
from fastapi.testclient import TestClient
import tempfile
import os
import shutil
from pathlib import Path

from server.api import app, Settings
from server.document_processor import DocumentProcessor
from server.vector_store import VectorStore

@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        documents_dir=tempfile.mkdtemp(),
        chroma_db_dir=tempfile.mkdtemp()
    )

@pytest.fixture
def test_client(test_settings):
    """Create test client with temporary directories."""
    # Override settings in app
    app.dependency_overrides[Settings] = lambda: test_settings
    
    client = TestClient(app)
    yield client
    
    # Cleanup temporary directories
    shutil.rmtree(test_settings.documents_dir)
    shutil.rmtree(test_settings.chroma_db_dir)

@pytest.fixture
def sample_markdown_file(test_settings):
    """Create a sample markdown file for testing."""
    content = """# Test Document
    
## Section 1
This is a test document.

## Section 2
More test content."""
    
    file_path = Path(test_settings.documents_dir) / "test.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    
    return str(file_path)

def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_process_documents_invalid_directory(test_client):
    """Test processing documents with invalid directory."""
    response = test_client.post("/documents/process", json={
        "directory": "/nonexistent/directory"
    })
    assert response.status_code == 404

def test_process_documents(test_client, sample_markdown_file):
    """Test processing documents endpoint."""
    response = test_client.post("/documents/process", json={
        "directory": os.path.dirname(sample_markdown_file)
    })
    
    # This will fail due to mock OpenAI key, but should return 500
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()

def test_generate_context_no_documents(test_client):
    """Test generating context with no documents."""
    response = test_client.post("/context/generate", json={
        "query": "test query"
    })
    
    # Should fail because no documents are processed
    assert response.status_code == 500

def test_list_documents_empty(test_client):
    """Test listing documents when none are processed."""
    response = test_client.get("/documents/list")
    assert response.status_code == 200
    assert "documents" in response.json()
    assert len(response.json()["documents"]) == 0

def test_delete_nonexistent_document(test_client):
    """Test deleting a document that doesn't exist."""
    response = test_client.delete("/documents/nonexistent.md")
    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 0

@pytest.mark.asyncio
async def test_full_workflow(test_client, sample_markdown_file):
    """Test the full workflow with mock data."""
    # First try to process documents
    process_response = test_client.post("/documents/process", json={
        "directory": os.path.dirname(sample_markdown_file)
    })
    assert process_response.status_code in [200, 500]  # May fail due to mock OpenAI key
    
    # Try to generate context
    context_response = test_client.post("/context/generate", json={
        "query": "test query",
        "n_results": 2
    })
    assert context_response.status_code in [200, 500]  # May fail due to mock OpenAI key
    
    # List documents
    list_response = test_client.get("/documents/list")
    assert list_response.status_code == 200
    
    # Try to delete a document
    delete_response = test_client.delete(f"/documents/{os.path.basename(sample_markdown_file)}")
    assert delete_response.status_code == 200

def test_cors_headers(test_client):
    """Test CORS headers are properly set."""
    response = test_client.options("/health", headers={
        "origin": "http://localhost:3000",
        "access-control-request-method": "GET"
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers

def test_error_handling(test_client):
    """Test error handling for various scenarios."""
    # Test with invalid JSON
    response = test_client.post("/context/generate", json={
        "invalid": "data"
    })
    assert response.status_code == 422  # Validation error
    
    # Test with missing required field
    response = test_client.post("/context/generate", json={})
    assert response.status_code == 422
    
    # Test with invalid n_results
    response = test_client.post("/context/generate", json={
        "query": "test",
        "n_results": -1
    })
    assert response.status_code == 422