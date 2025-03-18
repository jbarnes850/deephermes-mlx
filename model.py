"""
Model loading and configuration module for DeepHermes-3-Mistral-24B MLX inference.
"""
from typing import Tuple, Dict, Any, Optional

from mlx_lm import load
from transformers import PreTrainedTokenizer
import mlx.core as mx


def load_model(
    model_path: str = "Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16",
    trust_remote_code: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, PreTrainedTokenizer]:
    """
    Load the DeepHermes-3-Mistral-24B model and tokenizer.
    
    Args:
        model_path: Path to the model on Hugging Face Hub
        trust_remote_code: Whether to trust remote code in the tokenizer
        tokenizer_config: Additional tokenizer configuration
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    if tokenizer_config is None:
        tokenizer_config = {}
    
    # Load the model and tokenizer
    model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)
    
    print("Model loaded successfully!")
    return model, tokenizer
