#!/usr/bin/env python3
"""
Test script for model export functionality.
"""
import os
import argparse
from pathlib import Path
import logging

from deephermes.core import load_model
from deephermes.export import quantize_model, save_quantized_model, validate_model

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the model export functionality."""
    parser = argparse.ArgumentParser(description="Test model export functionality")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="./exported_model", help="Output directory")
    parser.add_argument("--quantize", choices=["int8", "int4", "fp16", "none"], 
                        default="int8", help="Quantization precision")
    parser.add_argument("--format", choices=["mlx", "gguf"], 
                        default="mlx", help="Output format")
    parser.add_argument("--validate", action="store_true", help="Validate the exported model")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {args.model}")
    model, _ = load_model(args.model)
    
    # Skip quantization if none is selected
    if args.quantize != "none":
        logger.info(f"Quantizing model to {args.quantize}")
        model = quantize_model(model, precision=args.quantize)
    
    logger.info(f"Saving model to {output_dir} in {args.format} format")
    output_path = save_quantized_model(model, output_dir, format=args.format)
    
    if args.validate:
        logger.info("Validating exported model")
        is_valid, validation_results = validate_model(output_path)
        if is_valid:
            logger.info("Model validation successful!")
            logger.info(f"Validation results: {validation_results}")
        else:
            logger.error("Model validation failed!")
            logger.error(f"Validation results: {validation_results}")
    
    logger.info(f"Model export completed. Model saved to {output_path}")

if __name__ == "__main__":
    main()
