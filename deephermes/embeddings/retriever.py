"""
Retrieval logic for finding relevant documents.

This module provides functionality for retrieving relevant documents
from a vector store based on a query, with support for various retrieval strategies.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import re
import numpy as np

from deephermes.embeddings.vector_store import VectorStore

class Retriever:
    """Retriever for finding relevant documents in a vector store.
    
    This class provides an interface for retrieving relevant documents
    from a vector store based on a query, with support for various retrieval
    strategies and reranking.
    """
    
    def __init__(self, 
                vector_store: VectorStore,
                top_k: int = 5,
                strategy: str = "semantic"):
        """Initialize retriever with specified vector store and parameters.
        
        Args:
            vector_store: Vector store to retrieve documents from
            top_k: Number of documents to retrieve
            strategy: Retrieval strategy ('semantic', 'keyword', 'hybrid')
        """
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Define retrieval strategies
        self.strategies = {
            "semantic": self._retrieve_semantic,
            "keyword": self._retrieve_keyword,
            "hybrid": self._retrieve_hybrid
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Invalid retrieval strategy: {strategy}. "
                           f"Choose from: {list(self.strategies.keys())}")
        
        self.strategy = strategy
        self.retrieve_function = self.strategies[strategy]
    
    def retrieve(self, 
                query: str, 
                filter_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query.
        
        Args:
            query: Query text to search for
            filter_str: Optional filter string in SQL WHERE clause format
            
        Returns:
            List of relevant documents
        """
        return self.retrieve_function(query, filter_str)
    
    def _retrieve_semantic(self, 
                         query: str, 
                         filter_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using semantic search.
        
        Args:
            query: Query text to search for
            filter_str: Optional filter string in SQL WHERE clause format
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.search(
            query=query,
            filter_str=filter_str,
            limit=self.top_k
        )
    
    def _retrieve_keyword(self, 
                        query: str, 
                        filter_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using keyword search.
        
        Args:
            query: Query text to search for
            filter_str: Optional filter string in SQL WHERE clause format
            
        Returns:
            List of relevant documents
        """
        # Extract keywords from query (simple approach)
        keywords = re.findall(r'\b\w+\b', query.lower())
        
        # Build filter string for keyword search
        keyword_filters = []
        for keyword in keywords:
            if len(keyword) > 3:  # Skip short words
                keyword_filters.append(f"text LIKE '%{keyword}%'")
        
        # Combine with existing filter
        if keyword_filters:
            keyword_filter_str = " OR ".join(keyword_filters)
            if filter_str:
                filter_str = f"({filter_str}) AND ({keyword_filter_str})"
            else:
                filter_str = keyword_filter_str
        
        # Fall back to semantic search if no keywords
        if not keyword_filters:
            return self._retrieve_semantic(query, filter_str)
        
        # Use a generic embedding for ordering
        results = self.vector_store.search(
            query=query,
            filter_str=filter_str,
            limit=self.top_k
        )
        
        return results
    
    def _retrieve_hybrid(self, 
                       query: str, 
                       filter_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid search (semantic + keyword).
        
        Args:
            query: Query text to search for
            filter_str: Optional filter string in SQL WHERE clause format
            
        Returns:
            List of relevant documents
        """
        # Get results from both methods
        semantic_results = self._retrieve_semantic(query, filter_str)
        keyword_results = self._retrieve_keyword(query, filter_str)
        
        # Combine results, removing duplicates
        combined_results = {}
        
        # Add semantic results with higher weight
        for result in semantic_results:
            doc_id = result.get('id')
            if doc_id:
                result['score'] = result.get('_distance', 0) * 0.7  # Lower distance is better
                combined_results[doc_id] = result
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result.get('id')
            if doc_id and doc_id not in combined_results:
                result['score'] = result.get('_distance', 0) * 0.3  # Lower distance is better
                combined_results[doc_id] = result
        
        # Sort by score (lower is better)
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.get('score', float('inf'))
        )
        
        # Return top_k results
        return sorted_results[:self.top_k]
    
    def rerank(self, 
              query: str, 
              results: List[Dict[str, Any]],
              reranker: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Rerank retrieved documents based on the query.
        
        Args:
            query: Original query text
            results: List of retrieved documents
            reranker: Optional custom reranking function
            
        Returns:
            Reranked list of documents
        """
        if not results:
            return []
        
        if reranker is not None:
            # Use custom reranker
            return reranker(query, results)
        
        # Default reranking: use simple keyword matching to boost scores
        keywords = re.findall(r'\b\w+\b', query.lower())
        significant_keywords = [k for k in keywords if len(k) > 3]
        
        for result in results:
            text = result.get('text', '').lower()
            keyword_matches = sum(1 for k in significant_keywords if k in text)
            
            # Adjust score based on keyword matches (lower is better)
            original_score = result.get('_distance', 0)
            result['score'] = original_score * (1.0 - 0.1 * keyword_matches)
        
        # Sort by adjusted score (lower is better)
        reranked_results = sorted(
            results,
            key=lambda x: x.get('score', float('inf'))
        )
        
        return reranked_results
    
    def set_strategy(self, strategy: str) -> None:
        """Change the retrieval strategy.
        
        Args:
            strategy: New retrieval strategy
            
        Raises:
            ValueError: If an invalid strategy is specified
        """
        if strategy not in self.strategies:
            raise ValueError(f"Invalid retrieval strategy: {strategy}. "
                           f"Choose from: {list(self.strategies.keys())}")
        
        self.strategy = strategy
        self.retrieve_function = self.strategies[strategy]
