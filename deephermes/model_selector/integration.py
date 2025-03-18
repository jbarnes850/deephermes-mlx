"""
Integration module for the Model Selector with the chat interface.
"""
import argparse
from typing import Dict, Any, Optional

from deephermes.model_selector.hardware_detection import get_hardware_info
from deephermes.model_selector.model_recommender import recommend_model


def get_optimal_configuration() -> Dict[str, Any]:
    """
    Get the optimal model configuration based on hardware capabilities.
    
    Returns:
        Dictionary with optimal configuration (model, quantize, lazy_load)
    """
    # Get system specifications
    hardware_info = get_hardware_info()
    
    # Get model recommendation
    recommendation = recommend_model(hardware_info)
    
    # Convert to a simple dictionary for easy integration
    config = {
        'model': recommendation.model_config.model_id,
        'quantize': recommendation.model_config.quantization,
        'lazy_load': recommendation.model_config.lazy_load,
        'max_tokens': recommendation.model_config.max_tokens,
        'reasoning': True  # Enable reasoning by default for DeepHermes
    }
    
    return config


def get_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Update command-line arguments with optimal configuration.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Updated arguments
    """
    # Get optimal configuration
    config = get_optimal_configuration()
    
    # Update args if not already specified
    if not args.model and 'model' in config:
        args.model = config['model']
    
    if not args.quantize and 'quantize' in config:
        args.quantize = config['quantize']
    
    if not args.lazy_load and config.get('lazy_load', False):
        args.lazy_load = True
    
    return args


if __name__ == "__main__":
    # Example usage
    class Args:
        model = None
        quantize = None
        lazy_load = False
    
    args = Args()
    updated_args = get_model_args(args)
    
    print("Model Arguments:")
    print(f"Model ID: {updated_args.model}")
    print(f"Quantization: {updated_args.quantize}")
    print(f"Lazy Load: {updated_args.lazy_load}")
