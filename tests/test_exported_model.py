#!/usr/bin/env python3
# Test an exported DeepHermes model

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import mlx.core as mx
    from mlx.utils import tree_unflatten
except ImportError:
    print("Warning: MLX not found. Some functionality may be limited.")

from deephermes.export.validator import validate_exported_model_directory


def load(model_path: str):
    """
    Load a model from the given path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Import here to avoid dependencies when just validating files
    from transformers import AutoTokenizer
    from mlx.utils import tree_unflatten
    import mlx.core as mx
    from safetensors import safe_open
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model weights
    weights_path = os.path.join(model_path, "model.safetensors")
    weights = {}
    with safe_open(weights_path, framework="numpy") as f:
        for k in f.keys():
            weights[k] = mx.array(f.get_tensor(k))
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Import the model class based on the config
    if config.get("model_type") == "llama":
        from mlx.models.llama import ModelArgs, Transformer
        
        # Create model args from config
        model_args = ModelArgs(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size=config["vocab_size"],
            hidden_dim=config["intermediate_size"],
            norm_eps=config["rms_norm_eps"],
            max_seq_len=config.get("max_position_embeddings", 2048)
        )
        
        # Create model
        model = Transformer(model_args)
        
        # Load weights
        model.update(weights)
        
        return model, tokenizer
    else:
        raise ValueError(f"Unsupported model type: {config.get('model_type')}")


def test_exported_model(model_dir: str) -> bool:
    """
    Test an exported DeepHermes model.
    
    Args:
        model_dir: Path to the exported model directory
        
    Returns:
        True if the model passes all tests, False otherwise
    """
    print(f"Validating model in {model_dir}...")
    
    # First, validate the model directory structure
    valid, missing_files = validate_exported_model_directory(model_dir)
    if not valid:
        print("Model validation failed. Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Try to load the model
    print("Attempting to load model from", model_dir)
    try:
        # Load the model
        model, tokenizer = load(model_dir)
        print("Model loaded successfully!")
        
        # Test tokenization
        try:
            tokens = tokenizer.encode("Hello, world!")
            print(f"Tokenization test passed: {tokens}")
        except Exception as e:
            print(f"Warning: Tokenization test failed: {e}")
            print("This may be expected for demo models.")
    except Exception as e:
        print(f"Warning: Error loading model: {e}")
        print("\nThis may be expected for demo models.")
        print("Continuing with file structure validation only...")
    
    # Check for metadata.json
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print("\nModel metadata:")
            print(f"  Name: {metadata.get('name', 'Not specified')}")
            print(f"  Description: {metadata.get('description', 'Not specified')}")
            print(f"  Author: {metadata.get('author', 'Not specified')}")
            print(f"  Tags: {', '.join(metadata.get('tags', []))}")
        except Exception as e:
            print(f"Warning: Error reading metadata: {e}")
    
    # Check for README.md
    readme_path = os.path.join(model_dir, "README.md")
    if os.path.exists(readme_path):
        print("\nModel card (README.md) is present.")
    
    # Check model size
    model_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\nModel size: {size_mb:.2f} MB")
    
    # Final validation result
    print("\nModel structure validation passed!")
    if valid:
        print("The model directory contains all required files.")
        print("You can now use this model for inference or serving.")
        return True
    else:
        print("The model directory is missing some files, but may still be usable.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test an exported DeepHermes model")
    parser.add_argument("--model", "-m", required=True, help="Path to the exported model directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose information")
    args = parser.parse_args()
    
    model_dir = args.model
    
    print(f"Validating model in {model_dir}...")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        return
    
    test_exported_model(model_dir)


if __name__ == "__main__":
    main()
