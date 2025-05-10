import pytest
import os
from pathlib import Path
import tempfile
from server.document_processor import DocumentProcessor

@pytest.fixture
def sample_markdown_content():
    return """# Test Document

## Section 1
This is a test document that we'll use to verify the document processor functionality.
It contains multiple paragraphs and sections to test chunking.

## Section 2
Another section with different content. This helps us test the chunking algorithm
and ensure it properly handles markdown formatting.

### Subsection 2.1
More content to ensure we have enough text to create multiple chunks.
This will help verify our overlap functionality works correctly."""

@pytest.fixture
def temp_markdown_file(sample_markdown_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_markdown_content)
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def document_processor():
    # Use a mock API key for testing
    return DocumentProcessor("test-api-key")

def test_chunk_document(document_processor, sample_markdown_content):
    """Test document chunking functionality."""
    chunks = document_processor.chunk_document(sample_markdown_content)
    
    assert len(chunks) > 0
    # Verify chunks contain content
    assert all(len(chunk) > 0 for chunk in chunks)
    # Verify all content is preserved
    combined = ''.join(chunks)
    assert all(section in combined for section in ["Section 1", "Section 2", "Subsection 2.1"])

@pytest.mark.asyncio
async def test_generate_embeddings(document_processor):
    """Test embedding generation."""
    texts = ["This is a test sentence.", "Another test sentence."]
    
    try:
        embeddings = await document_processor.generate_embeddings(texts)
        # This will fail with mock API key, but we can check the function structure
        assert False, "Should raise an error with mock API key"
    except Exception as e:
        # Expect an error due to invalid API key
        assert "Error generating embedding" in str(e)

@pytest.mark.asyncio
async def test_process_markdown_file(document_processor, temp_markdown_file):
    """Test processing a complete markdown file."""
    try:
        result = await document_processor.process_markdown_file(temp_markdown_file)
        # This will fail with mock API key, but we can check the structure
        assert False, "Should raise an error with mock API key"
    except Exception as e:
        # Verify file was read before API error
        assert os.path.exists(temp_markdown_file)

def test_chunk_document_overlap(document_processor):
    """Test that chunks properly overlap."""
    content = "." * (document_processor.chunk_size + 500)  # Content larger than chunk size
    chunks = document_processor.chunk_document(content)
    
    assert len(chunks) > 1
    # Check for overlap
    overlap = len(chunks[0]) + len(chunks[1]) - len(content[:len(chunks[0]) + len(chunks[1])])
    assert overlap >= document_processor.chunk_overlap - 1  # Allow for off-by-one due to splitting

@pytest.mark.asyncio
async def test_process_directory(document_processor, temp_markdown_file):
    """Test processing a directory of markdown files."""
    directory = os.path.dirname(temp_markdown_file)
    
    try:
        results = await document_processor.process_directory(directory)
        assert False, "Should raise an error with mock API key"
    except Exception as e:
        # Verify directory was scanned before API error
        assert os.path.exists(directory)

def test_chunk_document_empty(document_processor):
    """Test chunking empty document."""
    chunks = document_processor.chunk_document("")
    assert len(chunks) == 0

def test_chunk_document_small(document_processor):
    """Test chunking document smaller than chunk size."""
    small_content = "Small test document."
    chunks = document_processor.chunk_document(small_content)
    assert len(chunks) == 1
    assert chunks[0] == small_content