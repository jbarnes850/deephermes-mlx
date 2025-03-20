"""
DeepHermes MLX Local Data Module.

This module provides functionality for processing local files for use with
DeepHermes MLX, enabling both RAG (Retrieval-Augmented Generation) and 
fine-tuning on local data while maintaining privacy.
"""

from typing import List, Dict, Any, Optional, Union

from .file_processor import FileProcessor
from .chunker import TextChunker
from .formatters import RAGFormatter, FineTuningFormatter

__all__ = [
    'FileProcessor',
    'TextChunker',
    'RAGFormatter',
    'FineTuningFormatter',
]
