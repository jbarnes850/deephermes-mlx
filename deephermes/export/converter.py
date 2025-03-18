"""
Format conversion utilities for DeepHermes MLX models.

This module provides functions to convert models between different formats.
"""
from typing import Dict, Optional, Union, Literal, List, Any
import os
import json
import mlx.core as mx
import numpy as np
from pathlib import Path


def convert_to_mlx_format(
    model: Dict[str, Any],
    include_tokenizer: bool = True
) -> Dict[str, Any]:
    """
    Convert a model dictionary to MLX format.
    
    Args:
        model: Model dictionary
        include_tokenizer: Whether to include tokenizer in the output
        
    Returns:
        Model in MLX format
    """
    mlx_model = {}
    
    # Copy weights
    if "weights" in model:
        mlx_model["weights"] = model["weights"]
    elif "model" in model:
        mlx_model["weights"] = model["model"]
    
    # Copy config
    if "config" in model:
        mlx_model["config"] = model["config"]
    
    # Copy tokenizer if requested
    if include_tokenizer and "tokenizer" in model:
        mlx_model["tokenizer"] = model["tokenizer"]
    
    return mlx_model


def convert_to_gguf_format(
    model: Dict[str, Any],
    model_type: str = "llama"
) -> Dict[str, Any]:
    """
    Convert a model dictionary to GGUF format.
    
    Note: This is a placeholder implementation. Full GGUF support would require
    more complex logic and potentially external libraries.
    
    Args:
        model: Model dictionary
        model_type: Type of model architecture
        
    Returns:
        Model in GGUF format
    """
    # This is a placeholder for future implementation
    raise NotImplementedError("GGUF format conversion is not yet implemented")


def extract_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a model dictionary.
    
    Args:
        model: Model dictionary
        
    Returns:
        Dictionary of model metadata
    """
    metadata = {}
    
    # Extract basic model information
    if "config" in model:
        config = model["config"]
        metadata["model_type"] = config.get("model_type", "unknown")
        metadata["hidden_size"] = config.get("hidden_size", 0)
        metadata["num_attention_heads"] = config.get("num_attention_heads", 0)
        metadata["num_hidden_layers"] = config.get("num_hidden_layers", 0)
        metadata["vocab_size"] = config.get("vocab_size", 0)
    
    # Extract quantization information
    if "quantization" in model:
        metadata["quantization"] = model["quantization"]
    
    # Calculate model size
    weights = model.get("weights", model.get("model", {}))
    total_params = 0
    for name, weight in weights.items():
        if isinstance(weight, mx.array):
            total_params += weight.size
    
    metadata["total_parameters"] = total_params
    metadata["parameter_size_mb"] = sum(
        w.nbytes for w in weights.values() if isinstance(w, mx.array)
    ) / (1024 * 1024)
    
    return metadata


def generate_model_card(
    model: Dict[str, Any],
    model_name: str,
    description: str = "",
    author: str = "",
    license: str = "MIT",
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a model card for the exported model.
    
    Args:
        model: Model dictionary
        model_name: Name of the model
        description: Description of the model
        author: Author of the model
        license: License of the model
        additional_info: Additional information to include
        
    Returns:
        Model card as a string
    """
    metadata = extract_metadata(model)
    
    # Start building model card
    model_card = f"# {model_name}\n\n"
    
    if description:
        model_card += f"{description}\n\n"
    
    # Model information
    model_card += "## Model Information\n\n"
    model_card += f"- **Model Type:** {metadata.get('model_type', 'Unknown')}\n"
    model_card += f"- **Parameters:** {metadata.get('total_parameters', 0):,}\n"
    model_card += f"- **Size:** {metadata.get('parameter_size_mb', 0):.2f} MB\n"
    model_card += f"- **Hidden Size:** {metadata.get('hidden_size', 0)}\n"
    model_card += f"- **Attention Heads:** {metadata.get('num_attention_heads', 0)}\n"
    model_card += f"- **Layers:** {metadata.get('num_hidden_layers', 0)}\n\n"
    
    # Quantization information
    if "quantization" in metadata:
        quant = metadata["quantization"]
        model_card += "## Quantization\n\n"
        model_card += f"- **Precision:** {quant.get('precision', 'None')}\n"
        model_card += f"- **Block Size:** {quant.get('block_size', 0)}\n"
        
        if "original_size" in quant and "quantized_size" in quant:
            reduction = (1 - quant["quantized_size"] / quant["original_size"]) * 100
            model_card += f"- **Size Reduction:** {reduction:.2f}%\n\n"
    
    # Usage information
    model_card += "## Usage\n\n"
    model_card += "```python\n"
    model_card += "from deephermes.core.model import load_model\n"
    model_card += "from deephermes.core.inference import run_inference\n\n"
    model_card += f"model = load_model('{model_name}')\n"
    model_card += "response = run_inference(model, 'Your prompt here')\n"
    model_card += "print(response)\n"
    model_card += "```\n\n"
    
    # Additional information
    if additional_info:
        model_card += "## Additional Information\n\n"
        for key, value in additional_info.items():
            model_card += f"- **{key}:** {value}\n"
        model_card += "\n"
    
    # Author and license
    if author:
        model_card += f"## Author\n\n{author}\n\n"
    
    if license:
        model_card += f"## License\n\n{license}\n"
    
    return model_card


def save_model_card(
    model_card: str,
    output_dir: Union[str, Path]
) -> Path:
    """
    Save a model card to disk.
    
    Args:
        model_card: Model card as a string
        output_dir: Directory to save the model card
        
    Returns:
        Path to the saved model card
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_card_path = output_dir / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    return model_card_path
