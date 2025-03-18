"""
Quantization utilities for DeepHermes MLX models.

This module provides functions to quantize model weights for more efficient deployment.
"""
from typing import Dict, Optional, Union, Literal, List, Tuple
import os
import json
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path


def quantize_weights(
    weights: mx.array,
    precision: Literal["int8", "int4", "fp16"] = "int8",
    block_size: int = 32,
    calibration_data: Optional[np.ndarray] = None
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    Quantize individual weight tensor.
    
    Args:
        weights: Weight tensor to quantize
        precision: Target precision for quantization
        block_size: Block size for quantization
        calibration_data: Optional data for calibration-based quantization
            
    Returns:
        Quantized weights or tuple of (quantized_weights, scales)
    """
    if precision == "fp16":
        # Convert to float16
        return mx.array(weights.astype(mx.float16))
    
    elif precision == "int8":
        # Reshape to blocks
        original_shape = weights.shape
        weights_reshaped = weights.reshape(-1, block_size)
        
        # Compute scales (max absolute value in each block)
        scales = mx.max(mx.abs(weights_reshaped), axis=1, keepdims=True)
        scales = mx.maximum(scales, 1e-6)  # Avoid division by zero
        
        # Quantize to int8 range (-127 to 127)
        weights_quantized = mx.round((weights_reshaped / scales) * 127)
        weights_quantized = mx.clip(weights_quantized, -127, 127)
        weights_quantized = weights_quantized.astype(mx.int8)
        
        # Return quantized weights and scales
        return weights_quantized.reshape(original_shape), scales.reshape(-1, 1)
    
    elif precision == "int4":
        # Reshape to blocks
        original_shape = weights.shape
        weights_reshaped = weights.reshape(-1, block_size)
        
        # Compute scales (max absolute value in each block)
        scales = mx.max(mx.abs(weights_reshaped), axis=1, keepdims=True)
        scales = mx.maximum(scales, 1e-6)  # Avoid division by zero
        
        # Quantize to int4 range (-7 to 7)
        weights_quantized = mx.round((weights_reshaped / scales) * 7)
        weights_quantized = mx.clip(weights_quantized, -7, 7)
        
        # Pack two int4 values into one int8
        # This is a simplified approach - in production, we'd use bit packing
        weights_quantized = weights_quantized.astype(mx.int8)
        
        # Return quantized weights and scales
        return weights_quantized.reshape(original_shape), scales.reshape(-1, 1)
    
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def quantize_model(
    model: Dict,
    precision: Literal["int8", "int4", "fp16"] = "int8",
    block_size: int = 32,
    calibration_data: Optional[np.ndarray] = None
) -> Dict:
    """
    Quantize model weights to reduce size and improve inference speed.
    
    Args:
        model: Model dictionary with weights
        precision: Target precision for quantization
        block_size: Block size for quantization
        calibration_data: Optional data for calibration-based quantization
        
    Returns:
        Quantized model dictionary
    """
    quantized_model = {}
    
    # Copy non-weight items directly
    for key, value in model.items():
        if key not in ["weights", "model"]:
            quantized_model[key] = value
    
    # Get weights dictionary
    weights = model.get("weights", model.get("model", {}))
    quantized_weights = {}
    
    # Process each weight tensor
    for key, value in weights.items():
        if isinstance(value, mx.array) and value.dtype in [mx.float32, mx.float16]:
            # Quantize weight tensor
            if precision in ["int8", "int4"]:
                quantized_value, scales = quantize_weights(
                    value, precision, block_size, calibration_data
                )
                quantized_weights[key] = quantized_value
                quantized_weights[f"{key}_scales"] = scales
            else:
                quantized_weights[key] = quantize_weights(
                    value, precision, block_size, calibration_data
                )
        else:
            # Copy non-tensor or non-float values directly
            quantized_weights[key] = value
    
    # Add metadata about quantization
    quantized_model["quantization"] = {
        "precision": precision,
        "block_size": block_size,
        "original_size": sum(w.nbytes for w in weights.values() if isinstance(w, mx.array)),
        "quantized_size": sum(w.nbytes for w in quantized_weights.values() if isinstance(w, mx.array)),
    }
    
    # Store quantized weights
    quantized_model["weights"] = quantized_weights
    
    return quantized_model


def save_quantized_model(
    model: Dict,
    output_dir: Union[str, Path],
    format: Literal["mlx", "gguf"] = "mlx"
) -> Path:
    """
    Save a quantized model to disk.
    
    Args:
        model: Quantized model dictionary
        output_dir: Directory to save the model
        format: Output format (mlx or gguf)
        
    Returns:
        Path to the saved model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "mlx":
        # Save model weights
        weights_path = output_dir / "model.safetensors"
        mx.save_safetensors(str(weights_path), model["weights"])
        
        # Save model config and tokenizer config
        if "config" in model:
            with open(output_dir / "config.json", "w") as f:
                json.dump(model["config"], f, indent=2)
        
        if "tokenizer" in model:
            tokenizer = model["tokenizer"]
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(output_dir)
            else:
                with open(output_dir / "tokenizer_config.json", "w") as f:
                    json.dump(tokenizer, f, indent=2)
        
        # Save quantization metadata
        if "quantization" in model:
            with open(output_dir / "quantization.json", "w") as f:
                json.dump(model["quantization"], f, indent=2)
        
        return output_dir
    
    elif format == "gguf":
        # GGUF format support would be implemented here
        # This is a placeholder for future implementation
        raise NotImplementedError("GGUF format support is not yet implemented")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_adapter_with_base(
    adapter_weights: Dict,
    base_model: Dict,
    adapter_prefix: str = "lora_"
) -> Dict:
    """
    Merge LoRA adapter weights with base model weights.
    
    Args:
        adapter_weights: Dictionary of adapter weights
        base_model: Dictionary of base model weights
        adapter_prefix: Prefix used for adapter weight keys
        
    Returns:
        Dictionary with merged weights
    """
    merged_model = base_model.copy()
    merged_weights = merged_model.get("weights", merged_model.get("model", {})).copy()
    
    # Extract adapter A and B matrices
    adapter_a_matrices = {}
    adapter_b_matrices = {}
    
    for key, value in adapter_weights.items():
        if f"{adapter_prefix}A" in key:
            base_key = key.replace(f"{adapter_prefix}A", "")
            adapter_a_matrices[base_key] = value
        elif f"{adapter_prefix}B" in key:
            base_key = key.replace(f"{adapter_prefix}B", "")
            adapter_b_matrices[base_key] = value
    
    # Apply LoRA updates
    for base_key in adapter_a_matrices.keys():
        if base_key in adapter_b_matrices:
            if base_key in merged_weights:
                # Compute LoRA update: B * A
                a_matrix = adapter_a_matrices[base_key]
                b_matrix = adapter_b_matrices[base_key]
                update = mx.matmul(b_matrix, a_matrix)
                
                # Apply update to base weights
                merged_weights[base_key] = merged_weights[base_key] + update
    
    # Update merged model with merged weights
    if "weights" in merged_model:
        merged_model["weights"] = merged_weights
    else:
        merged_model["model"] = merged_weights
    
    return merged_model
