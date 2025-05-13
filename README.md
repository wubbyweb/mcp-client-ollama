# RAG-Enhanced Chat Application

A chat application that combines local document knowledge with Ollama's phi4 model using Retrieval-Augmented Generation (RAG).

## Features

- Web-based chat interface
- Local document processing with markdown support
- RAG capabilities using OpenAI embeddings and ChromaDB
- Real-time context retrieval and integration
- Persistent vector storage
- REST API for document management

## Prerequisites

- Python 3.8+
- Node.js (for serving the web interface)
- Ollama with phi4 model installed
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key and other configurations.

## Running the Application

1. Start the MCP server:
```bash
python -m server
```

2. Start Ollama with CORS enabled:
```bash
OLLAMA_ORIGINS=* ollama serve
```

3. Ensure phi4 model is installed:
```bash
ollama pull phi4
```
4. Open the application in your browser:
```
Open  index.html file
```

## Project Structure

```
.
├── server/
│   ├── __init__.py
│   ├── __main__.py
│   ├── api.py
│   ├── config.py
│   ├── document_processor.py
│   └── vector_store.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_document_processor.py
│   └── test_vector_store.py
├── documents/        # Directory for markdown files
├── app.js           # Web interface logic
├── index.html       # Web interface markup
├── styles.css       # Web interface styling
├── requirements.txt # Python dependencies
└── .env.example    # Environment variables template
```

## Testing

Run the test suite:
```bash
pytest tests/ -v --cov=server
```

## API Endpoints

- `GET /health` - Health check
- `POST /documents/process` - Process markdown files in a directory
- `POST /context/generate` - Generate context for a query
- `GET /documents/list` - List all processed documents
- `DELETE /documents/{source}` - Delete a document
- `DELETE /documents/clear` - Clear all documents from the vector store

## Usage Example

1. Place your markdown files in the `documents` directory.

2. Process the documents:
```bash
curl -X POST http://localhost:8000/documents/process \
     -H "Content-Type: application/json" \
     -d '{"directory": "./documents"}'
```

3. Open the web interface and start chatting. The application will:
   - Retrieve relevant context from your documents
   - Enhance Ollama's responses with this context
   - Display the source of the context used

## Development

- The server uses FastAPI for the backend API
- ChromaDB for vector storage
- OpenAI's text-embedding-ada-002 for embeddings
- Ollama's phi4 model for text generation
- Simple HTML/CSS/JS frontend

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project as you wish.

