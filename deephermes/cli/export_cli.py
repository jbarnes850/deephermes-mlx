"""
Command-line interface for exporting DeepHermes MLX models.

This module provides a CLI for exporting models for deployment.
"""
import argparse
import os
import sys
import json
import datetime
from typing import Dict, Optional, Union, Literal, List, Any, Tuple
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

from deephermes.export.quantize import quantize_weights, save_quantized_model
from deephermes.export.metadata import create_model_metadata, save_model_metadata, generate_model_card
from deephermes.export.validator import validate_exported_model_directory


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export DeepHermes MLX models")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model or adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base model if using adapter"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./exported_model",
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int8", "int4", "fp16", "none"],
        default="none",
        help="Quantization precision"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mlx"],
        default="mlx",
        help="Output format"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the exported model"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description for the exported model"
    )
    parser.add_argument(
        "--author",
        type=str,
        default="",
        help="Author of the exported model"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for the exported model"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the exported model"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with a minimal test model"
    )
    
    return parser.parse_args()


def is_lora_adapter(model_path: str) -> bool:
    """
    Check if the given path contains a LoRA adapter.
    
    Args:
        model_path: Path to check
        
    Returns:
        True if the path contains a LoRA adapter, False otherwise
    """
    path = Path(model_path)
    adapter_files = [
        "adapter_config.json",
        "adapters.safetensors",
        "0000100_adapters.safetensors",
        "adapter_model.safetensors"
    ]
    
    for file in adapter_files:
        if (path / file).exists():
            return True
    
    return False


def merge_lora_adapter(
    adapter_path: str,
    base_model_path: str,
    output_dir: str,
    quantize: str = "none",
    block_size: int = 32
) -> Dict[str, Any]:
    """
    Merge a LoRA adapter with a base model.
    
    Args:
        adapter_path: Path to the adapter
        base_model_path: Path to the base model
        output_dir: Output directory
        quantize: Quantization precision
        block_size: Block size for quantization
        
    Returns:
        Dictionary containing the merged model
    """
    print(f"Loading base model from {base_model_path}...")
    base_model, tokenizer = load(base_model_path)
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    adapter_path = Path(adapter_path)
    
    # Find adapter weights file
    adapter_weights_path = None
    for filename in ["adapters.safetensors", "0000100_adapters.safetensors", "adapter_model.safetensors"]:
        if (adapter_path / filename).exists():
            adapter_weights_path = adapter_path / filename
            break
    
    if not adapter_weights_path:
        raise ValueError(f"Could not find adapter weights in {adapter_path}")
    
    # Load adapter weights
    adapter_weights = mx.load(str(adapter_weights_path))
    print(f"Loaded adapter weights with {len(adapter_weights)} parameters")
    
    # Load adapter config to get scaling factor
    adapter_config = {}
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        
        # Extract LoRA parameters
        lora_alpha = adapter_config.get("lora_alpha", 1.0)
        lora_r = adapter_config.get("r", 8)
        scaling = lora_alpha / lora_r
        print(f"Using LoRA scaling factor: {scaling} (alpha={lora_alpha}, r={lora_r})")
    else:
        scaling = 0.125  # Default scaling
        print(f"No adapter config found. Using default scaling factor: {scaling}")
    
    # Apply LoRA adapters to the base model
    print("Applying LoRA adapters to base model...")
    
    # Get base model parameters
    base_params = base_model.parameters()
    
    # Find all LoRA A and B matrices
    lora_keys = {}
    for key in adapter_weights.keys():
        if ".lora_A" in key:
            base_key = key.replace(".lora_A.weight", "")
            if base_key not in lora_keys:
                lora_keys[base_key] = {"A": key}
        elif ".lora_B" in key:
            base_key = key.replace(".lora_B.weight", "")
            if base_key in lora_keys:
                lora_keys[base_key]["B"] = key
            else:
                lora_keys[base_key] = {"B": key}
    
    # Apply LoRA updates
    updated_count = 0
    for base_key, lora_pair in lora_keys.items():
        if "A" in lora_pair and "B" in lora_pair:
            target_key = f"{base_key}.weight"
            if target_key in base_params:
                # Get original weight
                original_weight = base_params[target_key]
                
                # Get LoRA matrices
                lora_a = adapter_weights[lora_pair["A"]]
                lora_b = adapter_weights[lora_pair["B"]]
                
                # Compute update: W + (B·A)·scaling
                update = mx.matmul(lora_b, lora_a) * scaling
                
                # Ensure shapes match
                if update.shape == original_weight.shape:
                    # Apply update
                    base_params[target_key] = original_weight + update
                    updated_count += 1
                else:
                    print(f"Warning: Shape mismatch for {target_key}. "
                          f"Original: {original_weight.shape}, Update: {update.shape}")
    
    print(f"Applied {updated_count} LoRA updates to base model")
    
    # Create model dictionary
    model_dict = {
        "weights": base_params,
        "config": base_model.config,
        "tokenizer": tokenizer
    }
    
    # Apply quantization if requested
    if quantize != "none":
        print(f"Quantizing model to {quantize}...")
        quantized_weights = {}
        original_size = 0
        quantized_size = 0
        
        for key, value in base_params.items():
            if isinstance(value, mx.array) and value.dtype in [mx.float32, mx.float16]:
                original_size += value.nbytes
                
                if quantize in ["int8", "int4"]:
                    quantized_value, scales = quantize_weights(
                        value, quantize, block_size
                    )
                    quantized_weights[key] = quantized_value
                    quantized_weights[f"{key}_scales"] = scales
                    quantized_size += quantized_value.nbytes + scales.nbytes
                else:  # fp16
                    quantized_value = quantize_weights(
                        value, quantize, block_size
                    )
                    quantized_weights[key] = quantized_value
                    quantized_size += quantized_value.nbytes
            else:
                quantized_weights[key] = value
                if isinstance(value, mx.array):
                    quantized_size += value.nbytes
        
        # Add quantization metadata
        model_dict["quantization"] = {
            "precision": quantize,
            "block_size": block_size,
            "original_size": original_size,
            "quantized_size": quantized_size,
        }
        
        # Update weights
        model_dict["weights"] = quantized_weights
    
    return model_dict


def export_regular_model(
    model_path: str,
    output_dir: str,
    quantize: str = "none",
    block_size: int = 32
) -> Dict[str, Any]:
    """
    Export a regular (non-adapter) model.
    
    Args:
        model_path: Path to the model
        output_dir: Output directory
        quantize: Quantization precision
        block_size: Block size for quantization
        
    Returns:
        Dictionary containing the exported model
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    
    # Create model dictionary
    model_dict = {
        "weights": model.parameters(),
        "config": model.config,
        "tokenizer": tokenizer
    }
    
    # Apply quantization if requested
    if quantize != "none":
        print(f"Quantizing model to {quantize}...")
        quantized_weights = {}
        original_size = 0
        quantized_size = 0
        
        for key, value in model.parameters().items():
            if isinstance(value, mx.array) and value.dtype in [mx.float32, mx.float16]:
                original_size += value.nbytes
                
                if quantize in ["int8", "int4"]:
                    quantized_value, scales = quantize_weights(
                        value, quantize, block_size
                    )
                    quantized_weights[key] = quantized_value
                    quantized_weights[f"{key}_scales"] = scales
                    quantized_size += quantized_value.nbytes + scales.nbytes
                else:  # fp16
                    quantized_value = quantize_weights(
                        value, quantize, block_size
                    )
                    quantized_weights[key] = quantized_value
                    quantized_size += quantized_value.nbytes
            else:
                quantized_weights[key] = value
                if isinstance(value, mx.array):
                    quantized_size += value.nbytes
        
        # Add quantization metadata
        model_dict["quantization"] = {
            "precision": quantize,
            "block_size": block_size,
            "original_size": original_size,
            "quantized_size": quantized_size,
        }
        
        # Update weights
        model_dict["weights"] = quantized_weights
    
    return model_dict


def create_demo_model() -> Dict[str, Any]:
    """
    Create a minimal test model for demonstration purposes.
    
    Returns:
        Dictionary containing a minimal model for demonstration
    """
    print("Creating demo model for export demonstration...")
    
    # Create minimal weights
    weights = {
        "model.embed_tokens.weight": mx.zeros((1024, 128)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((128, 128)),
        "model.layers.0.self_attn.k_proj.weight": mx.zeros((128, 128)),
        "model.layers.0.self_attn.v_proj.weight": mx.zeros((128, 128)),
        "model.layers.0.self_attn.o_proj.weight": mx.zeros((128, 128)),
        "model.layers.0.mlp.gate_proj.weight": mx.zeros((512, 128)),
        "model.layers.0.mlp.up_proj.weight": mx.zeros((512, 128)),
        "model.layers.0.mlp.down_proj.weight": mx.zeros((128, 512)),
        "model.layers.0.input_layernorm.weight": mx.zeros((128,)),
        "model.layers.0.post_attention_layernorm.weight": mx.zeros((128,)),
        "model.norm.weight": mx.zeros((128,)),
        "lm_head.weight": mx.zeros((1024, 128))
    }
    
    # Create minimal config
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "architectures": ["LlamaForCausalLM"]
    }
    
    # Create minimal tokenizer
    tokenizer = {
        "model_max_length": 2048,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "added_tokens": [],
        "special_tokens_map": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        }
    }
    
    return {
        "weights": weights,
        "config": config,
        "tokenizer": tokenizer
    }


def main():
    """Main entry point for export CLI."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model type and export
    model_dict = None
    
    if args.demo:
        print("Running in demo mode...")
        model_dict = create_demo_model()
    elif is_lora_adapter(args.model):
        print("Detected LoRA adapter, merging with base model...")
        if not args.base_model:
            print("Error: Base model is required for LoRA adapters.")
            print("Please specify a base model with --base-model.")
            sys.exit(1)
        
        model_dict = merge_lora_adapter(
            args.model,
            args.base_model,
            str(output_dir),
            args.quantize,
            args.block_size
        )
    else:
        print("Detected regular model, exporting...")
        try:
            model_dict = export_regular_model(
                args.model,
                str(output_dir),
                args.quantize,
                args.block_size
            )
        except Exception as e:
            if args.demo:
                print(f"Error loading model: {e}")
                print("Falling back to demo model...")
                model_dict = create_demo_model()
            else:
                print(f"Error exporting model: {e}")
                sys.exit(1)
    
    # Create metadata
    model_name = args.model_name or Path(args.model).name
    metadata = create_model_metadata(
        model_dict,
        model_name=model_name,
        description=args.description,
        author=args.author,
        tags=args.tags
    )
    
    # Add quantization info to metadata if applicable
    if args.quantize != "none" and "quantization" in model_dict:
        metadata["quantization"] = model_dict["quantization"]
    
    # Add timestamp
    metadata["created_at"] = datetime.datetime.now().isoformat()
    
    # Save model
    print(f"Saving model to {output_dir}...")
    
    # Save weights
    weights_path = output_dir / "model.safetensors"
    try:
        mx.save_safetensors(str(weights_path), model_dict["weights"])
        print(f"Saved model weights to {weights_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")
        # Create an empty file for demo purposes
        if args.demo:
            with open(weights_path, "w") as f:
                f.write("{}")
            print("Created placeholder weights file for demo")
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        if isinstance(model_dict["config"], dict):
            json.dump(model_dict["config"], f, indent=2)
        else:
            # If config is an object with __dict__ attribute
            config_dict = vars(model_dict["config"])
            json.dump(config_dict, f, indent=2)
    print(f"Saved model config to {config_path}")
    
    # Save tokenizer files
    tokenizer = model_dict["tokenizer"]
    if isinstance(tokenizer, dict):
        # Save tokenizer.json
        with open(output_dir / "tokenizer.json", "w") as f:
            json.dump(tokenizer, f, indent=2)
        
        # Save tokenizer_config.json
        with open(output_dir / "tokenizer_config.json", "w") as f:
            json.dump({"model_max_length": tokenizer.get("model_max_length", 2048)}, f, indent=2)
        
        # Save special_tokens_map.json
        with open(output_dir / "special_tokens_map.json", "w") as f:
            json.dump(tokenizer.get("special_tokens_map", {}), f, indent=2)
    else:
        # Save tokenizer files using tokenizer's save_pretrained method if available
        try:
            tokenizer.save_pretrained(str(output_dir))
        except Exception as e:
            print(f"Error saving tokenizer: {e}")
            # Create minimal tokenizer files for demo
            if args.demo:
                with open(output_dir / "tokenizer.json", "w") as f:
                    json.dump({"model_max_length": 2048}, f, indent=2)
                with open(output_dir / "tokenizer_config.json", "w") as f:
                    json.dump({"model_max_length": 2048}, f, indent=2)
                with open(output_dir / "special_tokens_map.json", "w") as f:
                    json.dump({}, f, indent=2)
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")
    
    # Generate model card
    model_card_path = output_dir / "README.md"
    model_card = generate_model_card(metadata)
    with open(model_card_path, "w") as f:
        f.write(model_card)
    print(f"Generated model card at {model_card_path}")
    
    # Validate exported model if requested
    if args.validate:
        print("\nValidating exported model...")
        is_valid, errors = validate_exported_model_directory(output_dir)
        
        if is_valid:
            print("Model validation successful!")
        else:
            print("Model validation failed with the following errors:")
            for error in errors:
                print(f"  - {error}")
    
    print(f"\nModel successfully exported to {output_dir}")


if __name__ == "__main__":
    main()
