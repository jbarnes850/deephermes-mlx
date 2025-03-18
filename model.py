"""
Model loading module for DeepHermes-3-Llama-3-8B MLX inference.
"""
from typing import Any, Dict, Optional, Tuple

from transformers import PreTrainedTokenizer
from mlx_lm import load


def load_model(
    model_path: str = "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
    quantize: Optional[str] = None,
    lazy_load: bool = False,
    trust_remote_code: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[Any, PreTrainedTokenizer]:
    """
    Load the model and tokenizer.
    
    Args:
        model_path: Path to the model (default: DeepHermes-3-Llama-3-8B-Preview-bf16)
        quantize: Quantization method to use (None, '4bit', or '8bit')
        lazy_load: Whether to load the model lazily to save memory
        trust_remote_code: Whether to trust remote code in the tokenizer
        tokenizer_config: Additional tokenizer configuration
        verbose: Whether to print verbose output
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Set up tokenizer config
    if tokenizer_config is None:
        tokenizer_config = {}
    
    tokenizer_config["trust_remote_code"] = trust_remote_code
    
    # Set up model config for quantization if specified
    model_config = {}
    if quantize:
        if quantize not in ['4bit', '8bit']:
            raise ValueError("Quantization must be either '4bit', '8bit', or None")
        
        # Add quantization to model config
        bits = 4 if quantize == '4bit' else 8
        model_config["quantization"] = {
            "group_size": 64,  # Standard group size for quantization
            "bits": bits
        }
    
    # Load the model and tokenizer
    model, tokenizer = load(
        path_or_hf_repo=model_path,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        lazy=lazy_load
    )
    
    if verbose:
        print("Model loaded successfully!")
    
    return model, tokenizer
