"""
Model loader for DeepHermes MLX.

This module provides functionality for loading models and tokenizers
from Hugging Face or local paths.
"""
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str
) -> Tuple[nn.Module, Any]:
    """
    Load a model and tokenizer from Hugging Face or a local path.
    
    Args:
        model_name_or_path: Model name on Hugging Face or path to local model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model and tokenizer from {model_name_or_path}")
    
    try:
        # Use mlx-lm for loading models, which is the recommended approach for MLX models
        from mlx_lm import load
        model, tokenizer = load(model_name_or_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model with mlx-lm: {e}")
        print("Falling back to manual loading...")
        
        # Check if path is local
        path = Path(model_name_or_path)
        if path.exists() and path.is_dir():
            # Load from local path
            model_path = path / "model.safetensors"
            config_path = path / "config.json"
            tokenizer_path = path
        else:
            # Download from Hugging Face
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=model_name_or_path, filename="model.safetensors")
                config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
                tokenizer_path = model_name_or_path
            except Exception as e:
                print(f"Error downloading model files: {e}")
                # Try alternative filenames
                try:
                    print("Trying alternative filenames...")
                    model_path = hf_hub_download(repo_id=model_name_or_path, filename="weights.safetensors")
                    config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
                    tokenizer_path = model_name_or_path
                except Exception as e:
                    print(f"Error downloading alternative model files: {e}")
                    raise ValueError(f"Could not download model files from {model_name_or_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model config and determine model type
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize model based on architecture
        model_type = config.get("model_type", "llama")
        if model_type == "llama":
            from mlx.models.llama import Llama, LlamaConfig
            model_config = LlamaConfig(**config)
            model = Llama(model_config)
        elif model_type == "mistral":
            from mlx.models.mistral import Mistral, MistralConfig
            model_config = MistralConfig(**config)
            model = Mistral(model_config)
        elif model_type == "phi":
            from mlx.models.phi import Phi, PhiConfig
            model_config = PhiConfig(**config)
            model = Phi(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load weights
        print(f"Loading weights from {model_path}")
        weights = mx.load(model_path)
        from mlx.utils import tree_unflatten
        model.update(tree_unflatten(list(weights.items())))
        
        return model, tokenizer
