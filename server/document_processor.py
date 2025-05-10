from typing import List, Dict, Any
import os
from pathlib import Path
import asyncio
from openai import AsyncOpenAI

class DocumentProcessor:
    def __init__(self, openai_api_key: str):
        """Initialize the document processor with OpenAI client."""
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def chunk_document(self, content: str) -> List[str]:
        """Split document content into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(content):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= len(content):
                chunks.append(content[start:])
                break
            
            # Find the last period or newline in chunk_size range
            last_period = content.rfind('.', start, end)
            last_newline = content.rfind('\n', start, end)
            split_point = max(last_period, last_newline)
            
            if split_point == -1 or split_point <= start:
                split_point = end
            
            chunks.append(content[start:split_point])
            start = split_point - self.chunk_overlap
            
        return chunks

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks using OpenAI."""
        embeddings = []
        
        for text in texts:
            try:
                response = await self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                raise
                
        return embeddings

    async def process_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Process a markdown file and return chunks with their embeddings."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate chunks
            chunks = self.chunk_document(content)
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(chunks)
            
            # Create metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "source": file_path,
                    "chunk_index": i,
                    "last_updated": str(Path(file_path).stat().st_mtime)
                })
            
            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise

    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all markdown files in a directory."""
        results = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    result = await self.process_markdown_file(file_path)
                    results.append(result)
        
        return results