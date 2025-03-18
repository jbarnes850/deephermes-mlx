#!/usr/bin/env python3
"""
MLX-LM Fine-tuning Script for Apple Silicon
This script provides a complete workflow for fine-tuning language models using MLX-LM on Apple Silicon.
"""

import os
import argparse
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Any
from datasets import load_dataset

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model with MLX-LM")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name or path (e.g., 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to store or load the dataset")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset name to use for fine-tuning")
    parser.add_argument("--adapter_path", type=str, default="./adapters",
                        help="Path to save the adapter weights")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                        help="Path to save the fused model")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to use from the dataset (for faster testing)")
    
    # Fine-tuning parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank of the LoRA adapter")
    parser.add_argument("--lora_alpha", type=float, default=16,
                        help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_layers", type=int, default=4,
                        help="Number of layers to apply LoRA to")
    parser.add_argument("--mask_prompt", action="store_true",
                        help="Whether to mask the prompt during training")
    
    # Workflow control
    parser.add_argument("--prepare_data", action="store_true",
                        help="Prepare the dataset for fine-tuning")
    parser.add_argument("--train", action="store_true",
                        help="Run the fine-tuning process")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the fine-tuned model")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse the adapter weights with the model")
    parser.add_argument("--generate", action="store_true",
                        help="Generate text with the fine-tuned model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for text generation")
    
    return parser.parse_args()

def prepare_data(args: argparse.Namespace) -> None:
    """Prepare the data for fine-tuning."""
    print(f"Preparing Alpaca data from {args.dataset_name}")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset(args.dataset_name)
    
    # Limit the number of examples if specified
    if args.num_examples > 0:
        dataset["train"] = dataset["train"].select(range(min(args.num_examples, len(dataset["train"]))))
    
    # Create train, validation, and test splits
    train_size = int(0.8 * len(dataset["train"]))
    val_size = int(0.1 * len(dataset["train"]))
    test_size = len(dataset["train"]) - train_size - val_size
    
    # Split the dataset
    splits = dataset["train"].train_test_split(
        train_size=train_size, test_size=val_size + test_size, seed=42
    )
    train_dataset = splits["train"]
    
    # Further split the test set into validation and test
    test_splits = splits["test"].train_test_split(
        train_size=val_size, test_size=test_size, seed=42
    )
    val_dataset = test_splits["train"]
    test_dataset = test_splits["test"]
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_path, exist_ok=True)
    
    # Process and save each split
    process_and_save_split(train_dataset, os.path.join(args.data_path, "train.jsonl"))
    process_and_save_split(val_dataset, os.path.join(args.data_path, "valid.jsonl"))
    process_and_save_split(test_dataset, os.path.join(args.data_path, "test.jsonl"))
    
    print(f"Data prepared and saved to {args.data_path}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")

def process_and_save_split(dataset: Any, output_file: str) -> None:
    """Process and save a dataset split in the format expected by MLX-LM."""
    with open(output_file, 'w') as f:
        for example in dataset:
            # Convert to the format expected by MLX-LM
            # Create a simple text format that doesn't require chat templates
            prompt = f"### Instruction:\n{example['instruction']}\n\n"
            if example.get('input', ''):
                prompt += f"### Input:\n{example['input']}\n\n"
            prompt += "### Response:"
            
            # Write to file in the format expected by MLX-LM
            json_line = json.dumps({"text": prompt + example['output']})
            f.write(json_line + '\n')

def run_fine_tuning(args: argparse.Namespace) -> None:
    """Run the fine-tuning process."""
    print(f"Fine-tuning model {args.model}")
    
    # Create adapter directory if it doesn't exist
    os.makedirs(args.adapter_path, exist_ok=True)
    
    # Construct the fine-tuning command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", args.data_path,
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", args.adapter_path
    ]
    
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    
    # Run the fine-tuning command
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)
    
    print(f"Fine-tuning completed. Adapter weights saved to {args.adapter_path}")

def evaluate_model(args: argparse.Namespace) -> None:
    """Evaluate the fine-tuned model."""
    print(f"Evaluating model {args.model} with adapter from {args.adapter_path}")
    
    # Construct the evaluation command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", args.model,
        "--adapter-path", args.adapter_path,
        "--data", args.data_path,
        "--test"
    ]
    
    # Run the evaluation command
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)
    
    print("Evaluation completed")

def fuse_model(args: argparse.Namespace) -> None:
    """Fuse the adapter weights with the model."""
    print(f"Fusing adapter weights with model {args.model}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find the adapter file
    adapter_files = list(Path(args.adapter_path).glob("*.safetensors"))
    if not adapter_files:
        adapter_files = list(Path(args.adapter_path).glob("*.npz"))
    
    if not adapter_files:
        print(f"No adapter files found in {args.adapter_path}")
        return
    
    adapter_file = str(adapter_files[0])
    
    # Construct the fuse command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", args.model,
        "--adapter-file", adapter_file,
        "--save-path", args.output_dir,
        "--fuse"
    ]
    
    # Run the fuse command
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)
    
    print(f"Model fused and saved to {args.output_dir}")

def run_generate(args: argparse.Namespace) -> None:
    """Generate text using the fine-tuned model."""
    if not args.prompt:
        print("No prompt provided. Please specify a prompt with --prompt.")
        return
        
    # Format the prompt to match our training data format
    formatted_prompt = f"### Instruction:\n{args.prompt}\n\n### Response:"
    
    # Determine whether to use the fused model or adapter
    model_path = args.output_dir if os.path.exists(args.output_dir) else args.model
    
    # Construct the generate command
    cmd = [
        "python", "-m", "mlx_lm.generate",
        "--model", model_path
    ]
    
    # Add adapter path if we're not using a fused model
    if model_path == args.model and os.path.exists(args.adapter_path):
        cmd.extend(["--adapter-path", args.adapter_path])
    
    # Add generation parameters
    cmd.extend([
        "--prompt", formatted_prompt,
        "--max-tokens", "200",
        "--temp", "0.7"
    ])
    
    # Run the generate command
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

def main() -> None:
    """Main function to run the fine-tuning workflow."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.adapter_path, exist_ok=True)
    
    # Run the requested steps
    if args.prepare_data and args.dataset_name:
        prepare_data(args)
    elif args.prepare_data and not args.dataset_name:
        print("Error: --dataset_name is required when using --prepare_data")
        return
    
    if args.train:
        run_fine_tuning(args)
    
    if args.evaluate:
        evaluate_model(args)
    
    if args.fuse:
        fuse_model(args)
    
    if args.generate:
        if not args.prompt:
            print("Warning: --prompt is required for text generation")
        else:
            run_generate(args)

if __name__ == "__main__":
    main()
