"""
Vector database for document storage and retrieval.

This module provides functionality for storing and retrieving document
embeddings using LanceDB, a lightweight vector database.
"""

from typing import List, Dict, Any, Optional, Union
import os
import numpy as np
from pathlib import Path
import json

from deephermes.embeddings.mlx_embeddings import MLXEmbeddings

class VectorStore:
    """LanceDB-based vector store for document storage and retrieval.
    
    This class provides an interface for storing and retrieving document
    embeddings using LanceDB, with support for filtering and metadata.
    """
    
    def __init__(self, 
                db_path: str, 
                embedding_model: str = "bge-small",
                table_name: str = "documents"):
        """Initialize vector store with specified path and embedding model.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model to use
            table_name: Name of the table to store documents in
            
        Raises:
            ImportError: If lancedb is not installed
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import lancedb
            self.db = lancedb.connect(str(self.db_path))
        except ImportError:
            raise ImportError(
                "lancedb is required for vector storage. "
                "Install it with 'pip install lancedb'"
            )
        
        self.table_name = table_name
        
        # Initialize embedding model
        self.embeddings = MLXEmbeddings(embedding_model)
        
        # Create or get table
        self._get_or_create_table()
    
    def _get_or_create_table(self):
        """Get existing table or create a new one.
        
        This method checks if the table exists and creates it if not.
        """
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
        else:
            # Create schema with sample data
            sample_data = [{
                "id": "sample",
                "text": "Sample document",
                "embedding": np.zeros(self.embeddings.dimensions).tolist(),
                "metadata": json.dumps({"source": "sample"})
            }]
            self.table = self.db.create_table(self.table_name, sample_data)
            # Remove sample data
            self.table.delete("id = 'sample'")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add, each with 'text' and optional 'metadata'
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Extract texts for embedding
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_texts(texts)
        
        # Prepare data for insertion
        data = []
        for i, doc in enumerate(documents):
            # Convert metadata to JSON string
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                metadata_str = json.dumps(metadata)
            else:
                metadata_str = json.dumps({})
            
            data.append({
                "id": doc.get("id", f"doc_{i}_{hash(doc['text'][:50])}"),
                "text": doc["text"],
                "embedding": embeddings[i].tolist(),
                "metadata": metadata_str
            })
        
        # Add to table
        self.table.add(data)
        return len(data)
    
    def search(self, 
              query: str, 
              filter_str: Optional[str] = None,
              limit: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.
        
        Args:
            query: Query text to search for
            filter_str: Optional filter string in SQL WHERE clause format
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Perform search
        search_query = self.table.search(query_embedding.tolist())
        
        if filter_str:
            search_query = search_query.where(filter_str)
        
        results = search_query.limit(limit).to_pandas()
        
        # Parse metadata JSON strings
        for i, row in results.iterrows():
            try:
                metadata = json.loads(row['metadata'])
                results.at[i, 'metadata'] = metadata
            except (json.JSONDecodeError, TypeError):
                results.at[i, 'metadata'] = {}
        
        # Convert to list of dictionaries
        return results.to_dict(orient="records")
    
    def delete_documents(self, filter_str: str) -> int:
        """Delete documents matching the filter.
        
        Args:
            filter_str: Filter string in SQL WHERE clause format
            
        Returns:
            Number of documents deleted
        """
        return self.table.delete(filter_str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics about the vector store
        """
        count = len(self.table)
        return {
            "document_count": count,
            "table_name": self.table_name,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimensions": self.embeddings.dimensions
        }
