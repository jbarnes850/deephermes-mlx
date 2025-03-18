"""
Model loading module for DeepHermes-3-Llama-3-8B MLX inference.
"""
from typing import Any, Dict, Optional, Tuple, List, Union
import os
import logging
from pathlib import Path

from transformers import PreTrainedTokenizer
from mlx_lm import load
from huggingface_hub import list_models, model_info, hf_hub_download

# Configure logging
logger = logging.getLogger(__name__)

# List of known MLX-compatible models
MLX_COMPATIBLE_MODELS = [
    "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
    "mlx-community/DeepHermes-2-Mistral-7B-bf16",
    "mlx-community/Mistral-Small-24B-Instruct-2501-bf16",
    # Add more known compatible models here
]


def check_model_compatibility(model_path: str) -> bool:
    """
    Check if a model is compatible with MLX.
    
    Args:
        model_path: Path or HF repo ID of the model
        
    Returns:
        True if compatible, False otherwise
    """
    # If it's a local path, check if it has the required files
    if os.path.exists(model_path):
        # Check for safetensors files
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        if not safetensors_files:
            logger.warning(f"No safetensors files found in {model_path}")
            return False
        return True
    
    # If it's a HF repo, check if it's in the known compatible list
    if model_path in MLX_COMPATIBLE_MODELS:
        return True
    
    # Try to get model info from HF
    try:
        info = model_info(model_path)
        
        # Check if model has MLX-specific tags
        if "mlx" in info.tags:
            # Double-check for safetensors files
            has_safetensors = False
            for file in info.siblings:
                if file.rfilename.endswith(".safetensors"):
                    has_safetensors = True
                    break
            
            if not has_safetensors:
                logger.warning(f"Model {model_path} has MLX tag but no safetensors files")
                return False
            
            return True
        
        # Check if model is from mlx-community and has safetensors files
        if model_path.startswith("mlx-community/"):
            has_safetensors = False
            has_mlx_weights = False
            
            for file in info.siblings:
                if file.rfilename.endswith(".safetensors"):
                    has_safetensors = True
                elif file.rfilename == "weights.npz":
                    has_mlx_weights = True
            
            if has_safetensors or has_mlx_weights:
                return True
            
            logger.warning(f"Model {model_path} is from mlx-community but has no MLX-compatible weight files")
            return False
        
        # For other models, check if they have MLX-specific files
        for file in info.siblings:
            if file.rfilename.endswith(".safetensors") or file.rfilename == "weights.npz":
                return True
        
        logger.warning(f"No MLX-compatible weight files found in {model_path}")
        return False
    except Exception as e:
        logger.warning(f"Error checking model compatibility: {e}")
        return False


def get_verified_compatible_models() -> List[str]:
    """
    Get a list of verified MLX-compatible models.
    
    Returns:
        List of model IDs
    """
    # These models have been verified to work with MLX
    verified_models = [
        "mlx-community/Mistral-7B-v0.1-bf16",
        "mlx-community/Mistral-7B-Instruct-v0.1-bf16",
        "mlx-community/Mistral-7B-Instruct-v0.2-bf16",
        "mlx-community/Mistral-Small-24B-Instruct-2501-bf16",
        "mlx-community/Llama-2-7b-chat-bf16",
        "mlx-community/Llama-2-13b-chat-bf16",
        "mlx-community/Llama-2-70b-chat-bf16",
        "mlx-community/Llama-3-8B-Instruct-bf16",
        "mlx-community/Llama-3-70B-Instruct-bf16",
        "mlx-community/Phi-3-mini-4k-instruct-bf16",
        "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
    ]
    
    return verified_models


def suggest_alternative_models() -> List[str]:
    """
    Suggest alternative MLX-compatible models.
    
    Returns:
        List of model IDs
    """
    # First try to get verified models
    verified_models = get_verified_compatible_models()
    
    # Return a subset of verified models
    if verified_models:
        return verified_models[:5]
    
    # If that fails, try to query HF API
    try:
        # Look for models with safetensors files
        compatible_models = []
        
        # First check mlx-community models with bf16 format (most likely to be compatible)
        models = list_models(author="mlx-community", limit=10)
        for m in models:
            if "bf16" in m.id.lower():
                try:
                    info = model_info(m.id)
                    for file in info.siblings:
                        if file.rfilename.endswith(".safetensors") or file.rfilename == "weights.npz":
                            compatible_models.append(m.id)
                            break
                except Exception:
                    # Skip if we can't get model info
                    pass
        
        if compatible_models:
            return compatible_models[:5]
        
        # Fallback to known compatible models
        return MLX_COMPATIBLE_MODELS[:5]
    except Exception:
        # Fallback to known compatible models
        return MLX_COMPATIBLE_MODELS[:5]


def load_model(
    model_path: str = "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
    quantize: Optional[str] = None,
    lazy_load: bool = False,
    trust_remote_code: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    force_load: bool = False,
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
        force_load: Whether to force loading even if compatibility check fails
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if verbose:
        logger.info(f"Loading model from {model_path}...")
    
    # Check model compatibility
    if not force_load and not check_model_compatibility(model_path):
        alternatives = suggest_alternative_models()
        alternatives_str = "\n- ".join([""] + alternatives)
        
        error_msg = (
            f"Model {model_path} may not be compatible with MLX. "
            f"Consider using one of these compatible models instead:{alternatives_str}\n\n"
            f"If you're sure this model is compatible, use force_load=True."
        )
        raise ValueError(error_msg)
    
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
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load(
            path_or_hf_repo=model_path,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            lazy=lazy_load
        )
        
        if verbose:
            logger.info("Model loaded successfully!")
        
        return model, tokenizer
    except FileNotFoundError as e:
        # Provide more helpful error message
        if "No safetensors found" in str(e):
            alternatives = suggest_alternative_models()
            alternatives_str = "\n- ".join([""] + alternatives)
            
            error_msg = (
                f"Model {model_path} doesn't have the required safetensors files for MLX. "
                f"This usually means the model wasn't converted to MLX format. "
                f"Consider using one of these compatible models instead:{alternatives_str}"
            )
            raise FileNotFoundError(error_msg) from e
        raise
    except Exception as e:
        # Add context to other errors
        raise Exception(f"Error loading model {model_path}: {str(e)}") from e
