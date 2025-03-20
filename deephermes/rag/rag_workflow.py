"""
RAG workflow implementation for DeepHermes MLX.

This module provides the core RAG (Retrieval-Augmented Generation) workflow
implementation, integrating local document processing, embeddings, and retrieval
with the existing workflow system.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import logging
from pathlib import Path

from ..embeddings.mlx_embeddings import MLXEmbeddings
from ..embeddings.vector_store import VectorStore
from ..embeddings.retriever import Retriever
from ..local_data.file_processor import FileProcessor
from ..local_data.chunker import TextChunker
from ..local_data.formatters import RAGFormatter
from .prompt_templates import (
    format_retrieved_context,
    format_augmented_prompt,
    format_no_context_prompt
)


class RAGWorkflow:
    """RAG workflow implementation for DeepHermes MLX.
    
    This class provides the core functionality for RAG workflows,
    including document processing, embedding generation, retrieval,
    and integration with the existing workflow system.
    """
    
    def __init__(self,
                 vector_store_path: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 use_adaptive_model_selection: bool = True,
                 max_chunks_to_retrieve: int = 5):
        """Initialize the RAG workflow.
        
        Args:
            vector_store_path: Path to the vector store directory
            embedding_model: Name of the embedding model to use
            use_adaptive_model_selection: Whether to adaptively select model based on available memory
            max_chunks_to_retrieve: Maximum number of chunks to retrieve
        """
        # Set up vector store path
        if vector_store_path is None:
            home_dir = os.path.expanduser("~")
            vector_store_path = os.path.join(home_dir, ".deephermes", "vector_store")
        os.makedirs(vector_store_path, exist_ok=True)
        self.vector_store_path = vector_store_path
        
        # Initialize embedding model
        self.embedding_model = embedding_model or "e5-mistral-7b"
        self.embeddings = MLXEmbeddings(
            model_name=self.embedding_model,
            use_adaptive_model_selection=use_adaptive_model_selection
        )
        
        # Initialize vector store and retriever
        self.vector_store = VectorStore(
            path=vector_store_path,
            embedding_dimensions=self.embeddings.dimensions
        )
        
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            top_k=max_chunks_to_retrieve
        )
        
        # Initialize document processing components
        self.file_processor = FileProcessor()
        self.text_chunker = TextChunker()
        self.rag_formatter = RAGFormatter()
        
        # Logging
        self.logger = logging.getLogger("deephermes.rag")
    
    def add_document(self, 
                     file_path: str, 
                     collection_name: str = "default",
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> int:
        """Add a document to the RAG system.
        
        Args:
            file_path: Path to the document file
            collection_name: Name of the collection to add the document to
            chunk_size: Size of chunks to split the document into
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks added to the vector store
        """
        try:
            # Process the file
            self.logger.info(f"Processing file: {file_path}")
            document_text = self.file_processor.process_file(file_path)
            
            # Split into chunks
            chunks = self.text_chunker.chunk_text(
                text=document_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Format chunks for RAG
            formatted_chunks = self.rag_formatter.format_chunks(
                chunks=chunks,
                metadata={"source_document": os.path.basename(file_path)}
            )
            
            # Add to vector store
            self.logger.info(f"Adding {len(formatted_chunks)} chunks to vector store")
            self.vector_store.add_documents(
                documents=formatted_chunks,
                collection_name=collection_name
            )
            
            return len(formatted_chunks)
            
        except Exception as e:
            self.logger.error(f"Error adding document {file_path}: {e}")
            raise
    
    def add_directory(self, 
                      directory_path: str, 
                      collection_name: str = "default",
                      recursive: bool = True,
                      file_extensions: Optional[List[str]] = None,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200) -> int:
        """Add all documents in a directory to the RAG system.
        
        Args:
            directory_path: Path to the directory
            collection_name: Name of the collection to add the documents to
            recursive: Whether to recursively process subdirectories
            file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
            chunk_size: Size of chunks to split the documents into
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks added to the vector store
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv']
        
        total_chunks = 0
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory {directory_path} does not exist or is not a directory")
        
        # Get all files in the directory
        if recursive:
            files = list(directory.glob('**/*'))
        else:
            files = list(directory.glob('*'))
        
        # Filter by extension
        files = [f for f in files if f.is_file() and f.suffix.lower() in file_extensions]
        
        # Process each file
        for file_path in files:
            try:
                chunks_added = self.add_document(
                    file_path=str(file_path),
                    collection_name=collection_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                total_chunks += chunks_added
                self.logger.info(f"Added {chunks_added} chunks from {file_path}")
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                # Continue with next file
        
        return total_chunks
    
    def query(self, 
              query_text: str, 
              collection_name: str = "default",
              top_k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Query the RAG system with a user query.
        
        Args:
            query_text: User query text
            collection_name: Name of the collection to query
            top_k: Maximum number of chunks to retrieve (overrides instance default)
            
        Returns:
            Tuple of (augmented_prompt, retrieved_documents)
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query_text,
            collection_name=collection_name,
            top_k=top_k
        )
        
        # Format context from retrieved documents
        context = format_retrieved_context(retrieved_docs)
        
        # Create augmented prompt
        if retrieved_docs:
            augmented_prompt = format_augmented_prompt(query_text, context)
        else:
            augmented_prompt = format_no_context_prompt(query_text)
        
        return augmented_prompt, retrieved_docs
    
    def augment_messages(self, 
                         messages: List[Dict[str, str]], 
                         collection_name: str = "default",
                         top_k: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Augment chat messages with retrieved context.
        
        Args:
            messages: List of chat messages
            collection_name: Name of the collection to query
            top_k: Maximum number of chunks to retrieve
            
        Returns:
            Tuple of (augmented_messages, retrieved_documents)
        """
        if not messages:
            return messages, []
        
        # Extract the last user message
        last_user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user_message = message.get("content", "")
                break
        
        if not last_user_message:
            return messages, []
        
        # Query the RAG system
        augmented_prompt, retrieved_docs = self.query(
            query_text=last_user_message,
            collection_name=collection_name,
            top_k=top_k
        )
        
        # Create a new messages list with the augmented user message
        augmented_messages = []
        for message in messages:
            if message.get("role") == "user" and message.get("content") == last_user_message:
                augmented_messages.append({"role": "user", "content": augmented_prompt})
            else:
                augmented_messages.append(message)
        
        return augmented_messages, retrieved_docs
    
    def list_collections(self) -> List[str]:
        """List all collections in the vector store.
        
        Returns:
            List of collection names
        """
        return self.vector_store.list_collections()
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if the collection was deleted, False otherwise
        """
        return self.vector_store.delete_collection(collection_name)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        return self.vector_store.get_collection_info(collection_name)
