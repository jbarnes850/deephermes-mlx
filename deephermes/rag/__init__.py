"""
DeepHermes MLX RAG Module.

This module provides functionality for Retrieval-Augmented Generation (RAG)
using local documents, enhancing model responses with relevant context.
"""

from typing import List, Dict, Any, Optional, Union

from .rag_workflow import RAGWorkflow
from .workflow_integration import RAGWorkflowRunner, RAG_WORKFLOW
from .prompt_templates import (
    get_rag_system_prompt,
    format_retrieved_context,
    format_augmented_prompt,
    format_no_context_prompt
)

__all__ = [
    'RAGWorkflow',
    'RAGWorkflowRunner',
    'RAG_WORKFLOW',
    'get_rag_system_prompt',
    'format_retrieved_context',
    'format_augmented_prompt',
    'format_no_context_prompt'
]
