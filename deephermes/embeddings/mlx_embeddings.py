"""
MLX-based embeddings for efficient vector generation on Apple Silicon.

This module provides functionality for generating embeddings from text
using MLX-optimized models, leveraging the efficiency of Apple Silicon.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import os
import logging
from pathlib import Path

class MLXEmbeddings:
    """MLX-based embeddings for efficient vector generation on Apple Silicon.
    
    This class provides an interface for generating embeddings from text
    using MLX-optimized models, with support for various embedding models
    including lightweight options and more powerful LLM-based embeddings.
    """
    
    available_models = {
        "e5-mistral-7b": {
            "dimensions": 4096,
            "model_id": "e5-mistral-7b-instruct-mlx",
            "type": "llm",
            "description": "Powerful LLM-based embeddings (7B parameters, requires 16GB+ RAM)",
            "memory_requirement": "high"
        },
        "bge-small": {
            "dimensions": 384,
            "model_id": "bge-small-en-v1.5",
            "type": "encoder",
            "description": "Lightweight embeddings suitable for most use cases",
            "memory_requirement": "low"
        },
        "bge-base": {
            "dimensions": 768,
            "model_id": "bge-base-en-v1.5",
            "type": "encoder",
            "description": "Medium-sized embeddings with good performance",
            "memory_requirement": "medium"
        },
        "e5-small": {
            "dimensions": 384,
            "model_id": "e5-small-v2",
            "type": "encoder",
            "description": "Lightweight E5 embeddings",
            "memory_requirement": "low"
        }
    }
    
    def __init__(self, 
                 model_name: str = "e5-mistral-7b", 
                 cache_dir: Optional[str] = None,
                 use_adaptive_model_selection: bool = True):
        """Initialize with the specified embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory to cache model weights (default: ~/.cache/mlx-deephermes)
            use_adaptive_model_selection: Whether to adaptively select model based on available memory
            
        Raises:
            ValueError: If the specified model is not available
            ImportError: If required dependencies are not installed
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.available_models.keys())}")
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/mlx-deephermes")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Adaptive model selection based on available memory
        if use_adaptive_model_selection and model_name == "e5-mistral-7b":
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                if available_memory_gb < 16:
                    logging.warning(
                        f"Insufficient memory for e5-mistral-7b model (available: {available_memory_gb:.1f}GB, recommended: 16GB+). "
                        "Falling back to bge-base model."
                    )
                    model_name = "bge-base"
            except ImportError:
                # If psutil is not available, proceed with requested model
                pass
        
        self.model_name = model_name
        self.model_info = self.available_models[model_name]
        self.dimensions = self.model_info["dimensions"]
        self.model_type = self.model_info["type"]
        
        if self.model_type == "encoder":
            self._initialize_encoder_model()
        elif self.model_type == "llm":
            self._initialize_llm_model(cache_dir)
    
    def _initialize_encoder_model(self):
        """Initialize a standard encoder-based embedding model."""
        try:
            from mlx_embedding_models.embedding import EmbeddingModel
            self.model = EmbeddingModel.from_registry(self.model_info["model_id"])
        except ImportError:
            raise ImportError(
                "mlx_embedding_models is required for embedding generation. "
                "Install it with 'pip install mlx-embedding-models'"
            )
    
    def _initialize_llm_model(self, cache_dir: str):
        """Initialize an LLM-based embedding model."""
        try:
            import mlx.core as mx
            from transformers import AutoTokenizer
            
            # Import mlx-llm for model loading
            try:
                from mlx_llm.model import create_model
            except ImportError:
                raise ImportError(
                    "mlx-llm is required for LLM-based embeddings. "
                    "Install it with 'pip install git+https://github.com/riccardomusmeci/mlx-llm.git'"
                )
            
            # Check if weights file exists or needs to be downloaded
            weights_path = os.path.join(cache_dir, f"{self.model_info['model_id']}.npz")
            if not os.path.exists(weights_path):
                logging.info(f"Downloading weights for {self.model_info['model_id']}...")
                # In a real implementation, you would add code to download the weights
                # For now, we'll raise an error with instructions
                raise FileNotFoundError(
                    f"Weights file not found at {weights_path}. Please download the weights from "
                    f"https://huggingface.co/mlx-community/{self.model_info['model_id']}/resolve/main/weights.npz "
                    f"and save to {weights_path}"
                )
            
            # Load the model and tokenizer
            self.llm_model = create_model(
                "e5-mistral-7b-instruct",
                weights_path=weights_path,
                strict=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
            self.mx = mx  # Store reference to mlx.core
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies for LLM-based embeddings not found: {e}. "
                "Install with 'pip install mlx transformers'"
            )
    
    def _get_detailed_instruct(self, query: str) -> str:
        """Format query with instruction for e5-mistral model.
        
        Args:
            query: The query text
            
        Returns:
            Formatted instruction text
        """
        task_description = "Represent this sentence for retrieval:"
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _last_token_pool(self, embeds: Any, attn_mask: Any) -> Any:
        """Extract embeddings from the last token of each sequence.
        
        Args:
            embeds: Token embeddings
            attn_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        mx = self.mx  # Use stored reference to mlx.core
        
        left_padding = (attn_mask[:, -1].sum() == attn_mask.shape[0])
        if left_padding:
            return embeds[:, -1]
        else:
            sequence_lengths = attn_mask.sum(axis=1) - 1
            batch_size = embeds.shape[0]
            return embeds[mx.arange(batch_size), sequence_lengths]
    
    def embed_texts_llm(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Generate embeddings using LLM-based model with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            Array of embeddings, shape (n_texts, dimensions)
        """
        if not texts:
            return np.array([])
        
        # Format texts with instruction
        formatted_texts = [self._get_detailed_instruct(text) for text in texts]
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i+batch_size]
            
            # Tokenize
            max_length = 4096
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True
            )
            
            # Add EOS token
            batch_dict['input_ids'] = [
                input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']
            ]
            
            # Pad sequences
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            
            # Convert to MLX arrays
            x = self.mx.array(batch_dict["input_ids"].tolist())
            attn_mask = self.mx.array(batch_dict["attention_mask"].tolist())
            
            # Generate embeddings
            embeds = self.llm_model.embed(x)
            self.mx.eval(embeds)
            
            # Pool and normalize
            pooled_embeds = np.array(self._last_token_pool(embeds, attn_mask))
            norm_den = np.linalg.norm(pooled_embeds, axis=-1, keepdims=True)
            normalized_embeds = pooled_embeds / norm_den
            
            all_embeddings.append(normalized_embeds)
        
        return np.vstack(all_embeddings)
    
    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for a list of texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once (model-dependent default if None)
            
        Returns:
            Array of embeddings, shape (n_texts, dimensions)
        """
        if not texts:
            return np.array([])
        
        # Use model-specific embedding method
        if self.model_type == "encoder":
            # For encoder models, use a larger batch size
            encoder_batch_size = 32 if batch_size is None else batch_size
            return self._embed_texts_encoder(texts, encoder_batch_size)
        elif self.model_type == "llm":
            # For LLM models, use a smaller batch size
            llm_batch_size = 4 if batch_size is None else batch_size
            return self.embed_texts_llm(texts, llm_batch_size)
    
    def _embed_texts_encoder(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using encoder-based model with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            Array of embeddings, shape (n_texts, dimensions)
        """
        # Process in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        if self.model_type == "encoder":
            return self.model.encode([query])[0]
        elif self.model_type == "llm":
            return self.embed_texts_llm([query])[0]
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available embedding models.
        
        Returns:
            Dictionary of available models with their information
        """
        return cls.available_models.copy()
    
    @classmethod
    def get_model_by_memory_requirement(cls, requirement: str = "low") -> str:
        """Get a suitable model name based on memory requirements.
        
        Args:
            requirement: Memory requirement level ('low', 'medium', or 'high')
            
        Returns:
            Name of a suitable model
        """
        for name, info in cls.available_models.items():
            if info.get("memory_requirement") == requirement:
                return name
        
        # Fallback to bge-small if no matching model found
        return "bge-small"
