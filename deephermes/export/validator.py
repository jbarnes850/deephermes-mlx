"""
Model validation utilities for DeepHermes MLX.

This module provides functions to validate exported models.
"""
from typing import Dict, Optional, Union, Literal, List, Any, Tuple
import os
import json
import mlx.core as mx
import numpy as np
from pathlib import Path


def validate_model_structure(model: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of a model dictionary.
    
    Args:
        model: Model dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required keys
    if "weights" not in model and "model" not in model:
        errors.append("Model is missing weights dictionary")
    
    # Get weights dictionary
    weights = model.get("weights", model.get("model", {}))
    
    # Check if weights dictionary is empty
    if not weights:
        errors.append("Model weights dictionary is empty")
    
    # Check if config is present
    if "config" not in model:
        errors.append("Model is missing config dictionary")
    
    return len(errors) == 0, errors


def validate_quantized_model(model: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a quantized model.
    
    Args:
        model: Quantized model dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # First validate basic structure
    is_valid, structure_errors = validate_model_structure(model)
    errors.extend(structure_errors)
    
    # Check for quantization metadata
    if "quantization" not in model:
        errors.append("Quantized model is missing quantization metadata")
    else:
        quant = model["quantization"]
        if "precision" not in quant:
            errors.append("Quantization metadata is missing precision")
        
        if quant.get("precision") not in ["int8", "int4", "fp16"]:
            errors.append(f"Unsupported quantization precision: {quant.get('precision')}")
    
    # Get weights dictionary
    weights = model.get("weights", model.get("model", {}))
    
    # Check for quantized weights and scales
    if quant.get("precision") in ["int8", "int4"]:
        for key, value in weights.items():
            if key.endswith("_scales"):
                # Skip scale tensors
                continue
                
            if isinstance(value, mx.array) and not key.endswith("_scales"):
                # Check if corresponding scale tensor exists
                scale_key = f"{key}_scales"
                if scale_key not in weights:
                    errors.append(f"Missing scale tensor for quantized weight: {key}")
    
    return len(errors) == 0, errors


def validate_model_with_sample_input(
    model: Dict[str, Any],
    sample_input: str = "Hello, world!",
    max_tokens: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Validate a model by running inference with a sample input.
    
    Args:
        model: Model dictionary to validate
        sample_input: Sample input text
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Import here to avoid circular imports
        from deephermes.core.model import load_model_from_dict
        from deephermes.core.inference import run_inference
        
        # Load model from dictionary
        loaded_model = load_model_from_dict(model)
        
        # Run inference
        _ = run_inference(
            loaded_model,
            sample_input,
            max_tokens=max_tokens,
            temperature=0.0  # Deterministic for validation
        )
        
        return True, None
    except Exception as e:
        return False, str(e)


def validate_exported_model_directory(
    model_dir: Union[str, Path]
) -> Tuple[bool, List[str]]:
    """
    Validate an exported model directory.
    
    Args:
        model_dir: Path to exported model directory
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    model_dir = Path(model_dir)
    errors = []
    
    # Check if directory exists
    if not model_dir.exists():
        errors.append(f"Model directory does not exist: {model_dir}")
        return False, errors
    
    # Check for required files
    required_files = ["model.safetensors", "config.json"]
    for file in required_files:
        if not (model_dir / file).exists():
            errors.append(f"Missing required file: {file}")
    
    # Check if weights file is empty
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists() and weights_path.stat().st_size == 0:
        errors.append("Model weights file is empty")
    
    # Check if config file is valid JSON
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            errors.append("Config file is not valid JSON")
    
    # Check for tokenizer files
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    tokenizer_found = False
    for file in tokenizer_files:
        if (model_dir / file).exists():
            tokenizer_found = True
            break
    
    if not tokenizer_found:
        errors.append("No tokenizer files found")
    
    # Check for quantization metadata if present
    quant_path = model_dir / "quantization.json"
    if quant_path.exists():
        try:
            with open(quant_path, "r") as f:
                quant = json.load(f)
                if "precision" not in quant:
                    errors.append("Quantization metadata is missing precision")
        except json.JSONDecodeError:
            errors.append("Quantization file is not valid JSON")
    
    # Check for metadata file
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        errors.append("Missing metadata.json file")
    else:
        try:
            with open(metadata_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            errors.append("Metadata file is not valid JSON")
    
    return len(errors) == 0, errors
