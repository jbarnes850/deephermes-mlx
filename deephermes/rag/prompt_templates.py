"""
RAG-specific prompt templates.

This module provides prompt templates for RAG (Retrieval-Augmented Generation)
workflows, including system prompts and formatting for retrieved context.
"""

from typing import List, Dict, Any, Optional, Union


def get_rag_system_prompt() -> str:
    """Get the default system prompt for RAG workflows.
    
    Returns:
        Default RAG system prompt
    """
    return """You are DeepHermes, an AI assistant with access to a knowledge base of local documents.
When answering questions, follow these guidelines:

1. Use the retrieved context to provide accurate, helpful responses
2. If the context doesn't contain the answer, say so clearly
3. Cite sources when appropriate using [Source: filename]
4. Maintain a helpful, informative tone
5. When reasoning through complex questions, break down your thinking step by step
6. Prioritize information from the retrieved context over general knowledge

Remember that your primary goal is to provide accurate information based on the retrieved documents."""


def format_retrieved_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved documents as context for the model.
    
    Args:
        results: List of retrieved documents with metadata
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant documents found."
    
    context_parts = []
    
    for i, doc in enumerate(results):
        # Extract source information from metadata
        metadata = doc.get('metadata', {})
        source = metadata.get('source_document', f"Document {i+1}")
        
        # Format the document with source citation
        context_parts.append(
            f"[Source: {source}]\n{doc.get('text', '')}"
        )
    
    return "\n\n".join(context_parts)


def format_augmented_prompt(query: str, context: str) -> str:
    """Format a prompt augmented with retrieved context.
    
    Args:
        query: Original user query
        context: Retrieved context
        
    Returns:
        Augmented prompt
    """
    return f"""I'll help you answer this question using the following information:

{context}

Based on the above information, please answer: {query}"""


def format_no_context_prompt(query: str) -> str:
    """Format a prompt when no relevant context is found.
    
    Args:
        query: Original user query
        
    Returns:
        Formatted prompt
    """
    return f"""I'll help you answer this question, but I don't have specific information about it in my local knowledge base.
I'll answer based on my general knowledge, but please note that I may not have the most up-to-date or specific information.

Question: {query}"""
