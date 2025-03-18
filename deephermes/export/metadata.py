"""
Model metadata utilities for DeepHermes MLX.

This module provides functions for handling model metadata and model cards.
"""
from typing import Dict, Optional, Union, Literal, List, Any
import os
import json
import time
from pathlib import Path
import mlx.core as mx


def create_model_metadata(
    model: Dict[str, Any],
    model_name: str,
    description: str = "",
    author: str = "",
    base_model: Optional[str] = None,
    adapter_type: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create metadata for a model.
    
    Args:
        model: Model dictionary
        model_name: Name of the model
        description: Description of the model
        author: Author of the model
        base_model: Base model name if this is an adapter
        adapter_type: Type of adapter if applicable (e.g., "lora")
        tags: List of tags for the model
        
    Returns:
        Dictionary of model metadata
    """
    # Extract model config
    config = model.get("config", {})
    
    # Calculate model size
    weights = model.get("weights", model.get("model", {}))
    total_params = 0
    for name, weight in weights.items():
        if isinstance(weight, mx.array):
            total_params += weight.size
    
    # Create metadata
    metadata = {
        "name": model_name,
        "description": description,
        "author": author,
        "created_at": int(time.time()),
        "model_type": config.get("model_type", "unknown"),
        "architecture": {
            "hidden_size": config.get("hidden_size", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "num_hidden_layers": config.get("num_hidden_layers", 0),
            "vocab_size": config.get("vocab_size", 0),
            "total_parameters": total_params,
            "parameter_size_mb": sum(
                w.nbytes for w in weights.values() if isinstance(w, mx.array)
            ) / (1024 * 1024)
        },
        "tags": tags or []
    }
    
    # Add quantization info if present
    if "quantization" in model:
        metadata["quantization"] = model["quantization"]
    
    # Add adapter info if applicable
    if base_model:
        metadata["base_model"] = base_model
        metadata["is_adapter"] = True
        if adapter_type:
            metadata["adapter_type"] = adapter_type
    else:
        metadata["is_adapter"] = False
    
    return metadata


def save_model_metadata(
    metadata: Dict[str, Any],
    output_dir: Union[str, Path]
) -> Path:
    """
    Save model metadata to disk.
    
    Args:
        metadata: Model metadata dictionary
        output_dir: Directory to save the metadata
        
    Returns:
        Path to the saved metadata file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def generate_model_card(
    metadata: Dict[str, Any],
    usage_examples: Optional[List[Dict[str, str]]] = None,
    license_type: str = "MIT"
) -> str:
    """
    Generate a model card from metadata.
    
    Args:
        metadata: Model metadata dictionary
        usage_examples: List of usage examples, each with 'title' and 'code'
        license_type: License type for the model
        
    Returns:
        Model card as a string
    """
    model_card = f"# {metadata['name']}\n\n"
    
    if metadata.get("description"):
        model_card += f"{metadata['description']}\n\n"
    
    # Model information
    model_card += "## Model Information\n\n"
    model_card += f"- **Model Type:** {metadata.get('model_type', 'Unknown')}\n"
    
    arch = metadata.get("architecture", {})
    model_card += f"- **Parameters:** {arch.get('total_parameters', 0):,}\n"
    model_card += f"- **Size:** {arch.get('parameter_size_mb', 0):.2f} MB\n"
    model_card += f"- **Hidden Size:** {arch.get('hidden_size', 0)}\n"
    model_card += f"- **Attention Heads:** {arch.get('num_attention_heads', 0)}\n"
    model_card += f"- **Layers:** {arch.get('num_hidden_layers', 0)}\n"
    
    # Base model information if adapter
    if metadata.get("is_adapter", False):
        model_card += f"- **Base Model:** {metadata.get('base_model', 'Unknown')}\n"
        if "adapter_type" in metadata:
            model_card += f"- **Adapter Type:** {metadata['adapter_type']}\n"
    
    model_card += "\n"
    
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
    
    # Default usage example
    model_card += "```python\n"
    model_card += "from deephermes.core.model import load_model\n"
    model_card += "from deephermes.core.inference import run_inference\n\n"
    model_card += f"model = load_model('{metadata['name']}')\n"
    model_card += "response = run_inference(model, 'Your prompt here')\n"
    model_card += "print(response)\n"
    model_card += "```\n\n"
    
    # Additional usage examples
    if usage_examples:
        model_card += "### Examples\n\n"
        for example in usage_examples:
            if "title" in example and "code" in example:
                model_card += f"#### {example['title']}\n\n"
                model_card += f"```python\n{example['code']}\n```\n\n"
    
    # Tags
    if metadata.get("tags"):
        model_card += "## Tags\n\n"
        tags = ", ".join(f"`{tag}`" for tag in metadata["tags"])
        model_card += f"{tags}\n\n"
    
    # Author and license
    if metadata.get("author"):
        model_card += f"## Author\n\n{metadata['author']}\n\n"
    
    if license_type:
        model_card += f"## License\n\n{license_type}\n"
    
    return model_card


def load_model_metadata(
    model_dir: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Load model metadata from disk.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Model metadata dictionary or None if not found
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
