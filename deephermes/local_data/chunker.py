"""
Text chunking strategies for local data processing.

This module provides functionality for splitting documents into smaller chunks
for use in RAG and fine-tuning, with various chunking strategies.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import re

class TextChunker:
    """Split documents into smaller chunks for embedding and retrieval.
    
    This class provides various strategies for chunking text documents,
    including fixed-size chunks, sentence-based chunks, and paragraph-based chunks.
    """
    
    def __init__(self, 
                chunk_size: int = 512, 
                chunk_overlap: int = 128,
                strategy: str = "fixed"):
        """Initialize text chunker with specified parameters.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph')
        
        Raises:
            ValueError: If an invalid strategy is specified
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define chunking strategies
        self.strategies = {
            "fixed": self._chunk_fixed,
            "sentence": self._chunk_by_sentence,
            "paragraph": self._chunk_by_paragraph
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Invalid chunking strategy: {strategy}. "
                           f"Choose from: {list(self.strategies.keys())}")
        
        self.strategy = strategy
        self.chunk_function = self.strategies[strategy]
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the selected strategy.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        return self.chunk_function(text)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into chunks, preserving metadata.
        
        Args:
            document: Document dictionary with 'content' and 'metadata' keys
            
        Returns:
            List of document chunks with metadata
        """
        if not isinstance(document, dict) or 'content' not in document:
            raise ValueError("Document must be a dictionary with a 'content' key")
        
        content = document['content']
        if not isinstance(content, str):
            # For non-text content (e.g., JSON, CSV), convert to string
            if isinstance(content, (list, dict)):
                import json
                content = json.dumps(content)
            else:
                content = str(content)
        
        chunks = self.chunk_text(content)
        
        # Create document chunks with metadata
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunks.append({
                "content": chunk,
                "metadata": {
                    **document.get('metadata', {}),
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "source_document": document.get('filename', 'unknown'),
                    "source_path": document.get('path', 'unknown')
                }
            })
        
        return doc_chunks
    
    def _chunk_fixed(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of fixed-size text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of specified size
            end = start + self.chunk_size
            
            # If this is not the first chunk, include overlap
            if start > 0:
                start = start - self.chunk_overlap
                end = start + self.chunk_size
            
            # Ensure we don't go beyond the text length
            if end > len(text):
                end = len(text)
            
            # Add chunk to list
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to next chunk position
            start = end
            
            # Break if we've reached the end of the text
            if end == len(text):
                break
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence-based text chunks
        """
        # Simple sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence
            else:
                # Add sentence to current chunk
                current_chunk += sentence + " "
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Split text into chunks at paragraph boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraph-based text chunks
        """
        # Split by paragraph (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # If adding this paragraph would exceed the chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + paragraph + "\n\n"
            else:
                # Add paragraph to current chunk
                current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def set_strategy(self, strategy: str) -> None:
        """Change the chunking strategy.
        
        Args:
            strategy: New chunking strategy
            
        Raises:
            ValueError: If an invalid strategy is specified
        """
        if strategy not in self.strategies:
            raise ValueError(f"Invalid chunking strategy: {strategy}. "
                           f"Choose from: {list(self.strategies.keys())}")
        
        self.strategy = strategy
        self.chunk_function = self.strategies[strategy]
