#!/usr/bin/env python
"""
Command-line interface for the DeepHermes Model Selector.

This module provides a simple CLI for getting model recommendations
and generating configuration files for DeepHermes models.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_selector.hardware_detection import get_hardware_info, save_hardware_info
from model_selector.model_recommender import recommend_model, print_recommendation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepHermes Model Selector - Get recommendations for optimal model configuration."
    )
    
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the recommended configuration to a config file"
    )
    
    parser.add_argument(
        "--config-path",
        default="model_config.json",
        help="Path to save the configuration file (default: model_config.json)"
    )
    
    parser.add_argument(
        "--prioritize",
        choices=["speed", "quality", "memory"],
        default=None,
        help="Prioritize a specific aspect in the recommendation"
    )
    
    parser.add_argument(
        "--save-hardware-info",
        action="store_true",
        help="Save hardware information to a file"
    )
    
    parser.add_argument(
        "--hardware-info-path",
        default="hardware_info.json",
        help="Path to save the hardware information (default: hardware_info.json)"
    )
    
    parser.add_argument(
        "--force-model-size",
        choices=["3B", "8B", "24B"],
        default=None,
        help="Force a specific model size regardless of hardware"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the recommendation in JSON format for programmatic use"
    )
    
    return parser.parse_args()


def save_config_file(config_path: str, model_config: Dict[str, Any]) -> None:
    """
    Save model configuration to a JSON file.
    
    Args:
        config_path: Path to save the configuration
        model_config: Model configuration dictionary
    """
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def main() -> None:
    """Run the Model Selector CLI."""
    args = parse_args()
    
    # Get hardware information
    hardware_info = get_hardware_info()
    
    # Print hardware information if not in JSON mode
    if not args.json:
        print("===== Hardware Information =====")
        print(f"Device: {hardware_info.device_name}")
        print(f"Chip: {hardware_info.chip_type}")
        print(f"Memory: {hardware_info.memory_gb:.1f} GB")
        print(f"CPU Cores: {hardware_info.cpu_cores}")
        
        if hardware_info.is_apple_silicon:
            print(f"GPU Cores: {hardware_info.gpu_cores}")
            print(f"Neural Engine Cores: {hardware_info.neural_engine_cores}")
    
    # Save hardware information if requested
    if args.save_hardware_info:
        save_hardware_info(args.hardware_info_path)
    
    # Get model recommendation
    recommendation = recommend_model(hardware_info)
    
    # Apply priority adjustments if specified
    if args.prioritize:
        config = recommendation.model_config
        if args.prioritize == "speed":
            print("\nPrioritizing speed in recommendation...")
            # Prefer smaller models with less quantization
            if "24B" in config.model_id:
                # Switch to 8B model for better speed
                from model_selector.model_recommender import DEEPHERMES_MODELS, ModelConfig
                config = ModelConfig(
                    model_id=DEEPHERMES_MODELS["8B"]["model_id"],
                    quantization=None,  # Full precision for quality
                    lazy_load=False,    # No lazy loading for speed
                    max_tokens=1024,
                    memory_required_gb=DEEPHERMES_MODELS["8B"]["memory_required"]["none"],
                    performance_score=DEEPHERMES_MODELS["8B"]["performance_score"],
                    reasoning_quality_score=DEEPHERMES_MODELS["8B"]["reasoning_quality_score"]
                )
                recommendation.model_config = config
                recommendation.reason = "The 8B model is recommended for faster performance while maintaining good reasoning quality. Full precision (no quantization) provides the highest quality output."
            elif "8B" in config.model_id and hardware_info.memory_gb < 16:
                # Switch to 3B model for even better speed on constrained systems
                from model_selector.model_recommender import DEEPHERMES_MODELS, ModelConfig
                config = ModelConfig(
                    model_id=DEEPHERMES_MODELS["3B"]["model_id"],
                    quantization=None,  # Full precision for quality
                    lazy_load=False,    # No lazy loading for speed
                    max_tokens=1024,
                    memory_required_gb=DEEPHERMES_MODELS["3B"]["memory_required"]["none"],
                    performance_score=DEEPHERMES_MODELS["3B"]["performance_score"],
                    reasoning_quality_score=DEEPHERMES_MODELS["3B"]["reasoning_quality_score"]
                )
                recommendation.model_config = config
                recommendation.reason = "The 3B model is recommended for maximum speed. Full precision (no quantization) provides the highest quality output."
        elif args.prioritize == "quality":
            print("\nPrioritizing quality in recommendation...")
            # Try to use the largest model possible
            if ("3B" in config.model_id or "8B" in config.model_id) and hardware_info.memory_gb >= 32:
                # Switch to 24B model for best quality on high-end systems
                from model_selector.model_recommender import DEEPHERMES_MODELS, ModelConfig
                config = ModelConfig(
                    model_id=DEEPHERMES_MODELS["24B"]["model_id"],
                    quantization=None if hardware_info.memory_gb >= 48 else "8bit",  # Use quantization if needed
                    lazy_load=hardware_info.memory_gb < 48,  # Use lazy loading if memory is constrained
                    max_tokens=1024,
                    memory_required_gb=DEEPHERMES_MODELS["24B"]["memory_required"]["none" if hardware_info.memory_gb >= 48 else "8bit"],
                    performance_score=DEEPHERMES_MODELS["24B"]["performance_score"],
                    reasoning_quality_score=DEEPHERMES_MODELS["24B"]["reasoning_quality_score"]
                )
                recommendation.model_config = config
                recommendation.reason = "The 24B model is recommended for best reasoning quality. " + \
                                      ("Full precision (no quantization) provides the highest quality output." 
                                       if hardware_info.memory_gb >= 48 else 
                                       "8-bit quantization helps reduce memory usage while maintaining good quality.")
            elif "3B" in config.model_id and hardware_info.memory_gb >= 16:
                # Switch to 8B model for better quality on mid-range systems
                from model_selector.model_recommender import DEEPHERMES_MODELS, ModelConfig
                config = ModelConfig(
                    model_id=DEEPHERMES_MODELS["8B"]["model_id"],
                    quantization=None,  # Full precision for quality
                    lazy_load=False,    # No lazy loading for quality
                    max_tokens=1024,
                    memory_required_gb=DEEPHERMES_MODELS["8B"]["memory_required"]["none"],
                    performance_score=DEEPHERMES_MODELS["8B"]["performance_score"],
                    reasoning_quality_score=DEEPHERMES_MODELS["8B"]["reasoning_quality_score"]
                )
                recommendation.model_config = config
                recommendation.reason = "The 8B model is recommended for better reasoning quality. Full precision (no quantization) provides the highest quality output."
        elif args.prioritize == "memory":
            print("\nPrioritizing memory efficiency in recommendation...")
            # Always use quantization and lazy loading
            config.quantization = "4bit"  # Most aggressive quantization
            config.lazy_load = True
            recommendation.reason = "Memory efficiency prioritized: Using 4-bit quantization and lazy loading to minimize memory usage while maintaining reasonable quality."
    
    # Force a specific model size if requested
    if args.force_model_size:
        from model_selector.model_recommender import DEEPHERMES_MODELS, ModelConfig
        size = args.force_model_size
        
        # Determine appropriate quantization based on available memory
        if size == "24B":
            if hardware_info.memory_gb >= 48:
                quant = None  # Full precision
            elif hardware_info.memory_gb >= 24:
                quant = "8bit"
            else:
                quant = "4bit"
        elif size == "8B":
            if hardware_info.memory_gb >= 16:
                quant = None  # Full precision
            elif hardware_info.memory_gb >= 8:
                quant = "8bit"
            else:
                quant = "4bit"
        else:  # 3B
            if hardware_info.memory_gb >= 6:
                quant = None  # Full precision
            elif hardware_info.memory_gb >= 3:
                quant = "8bit"
            else:
                quant = "4bit"
        
        # Determine if lazy loading is needed
        lazy = (size == "24B" and hardware_info.memory_gb < 48) or \
               (size == "8B" and hardware_info.memory_gb < 16) or \
               (size == "3B" and hardware_info.memory_gb < 6)
        
        # Create the forced model config
        config = ModelConfig(
            model_id=DEEPHERMES_MODELS[size]["model_id"],
            quantization=quant,
            lazy_load=lazy,
            max_tokens=1024,
            memory_required_gb=DEEPHERMES_MODELS[size]["memory_required"]["none" if quant is None else quant],
            performance_score=DEEPHERMES_MODELS[size]["performance_score"],
            reasoning_quality_score=DEEPHERMES_MODELS[size]["reasoning_quality_score"]
        )
        
        recommendation.model_config = config
        recommendation.reason = f"Forced {size} model as requested. " + \
                              (f"{quant} quantization" if quant else "Full precision") + \
                              (" with lazy loading" if lazy else "") + \
                              " configured based on available memory."
    
    # Print recommendation
    if args.json:
        config = recommendation.model_config
        output = {
            "model_id": config.model_id,
            "quantization": config.quantization,
            "lazy_load": config.lazy_load,
            "max_tokens": config.max_tokens,
            "reasoning": True,
            "reason": recommendation.reason,
            "confidence": recommendation.confidence
        }
        print(json.dumps(output))
    else:
        print_recommendation(recommendation)
    
    # Save configuration if requested
    if args.save_config:
        config = recommendation.model_config
        config_dict = {
            "model_id": config.model_id,
            "quantization": config.quantization,
            "lazy_load": config.lazy_load,
            "max_tokens": config.max_tokens,
            "reasoning": True
        }
        save_config_file(args.config_path, config_dict)


if __name__ == "__main__":
    main()
