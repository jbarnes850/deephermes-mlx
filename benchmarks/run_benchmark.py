#!/usr/bin/env python
"""
Command-line interface for running DeepHermes benchmarks.

This script provides a simple interface for running benchmarks on DeepHermes models,
with options for selecting models, datasets, and other benchmark parameters.
"""

import argparse
import sys
import os
from typing import List, Optional

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_runner import BenchmarkConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for DeepHermes models on reasoning tasks."
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlx-community/DeepHermes-3-Llama-3-3B-Preview-bf16",
            "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
            "mlx-community/DeepHermes-3-Mistral-24B-Preview-bf16"
        ],
        help="List of model IDs to benchmark"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mmlu", "gsm8k"],
        help="List of datasets to benchmark on"
    )
    
    # MMLU subset selection
    parser.add_argument(
        "--mmlu-subset",
        default="stem",
        choices=["stem", "humanities", "social_sciences", "other"],
        help="MMLU subset to use"
    )
    
    # Sample limits
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to use from each dataset (default: all)"
    )
    
    # Memory optimization
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantize models to reduce memory usage"
    )
    
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Load model weights lazily to reduce memory usage"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to save benchmark results"
    )
    
    # Reasoning mode
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning mode"
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the benchmark with the specified configuration."""
    args = parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        model_ids=args.models,
        dataset_names=args.datasets,
        batch_size=1,
        max_tokens=args.max_tokens,
        quantization=args.quantize,
        lazy_load=args.lazy_load,
        reasoning=not args.no_reasoning,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=================")
    
    for model_id in results:
        print(f"\nModel: {model_id}")
        
        for dataset_name in results[model_id]["datasets"]:
            metrics = results[model_id]["datasets"][dataset_name]["metrics"]
            memory_usage = results[model_id]["datasets"][dataset_name].get("memory_usage_mb", 0)
            total_time = results[model_id]["datasets"][dataset_name].get("total_time_seconds", 0)
            
            print(f"  Dataset: {dataset_name}")
            print(f"    Accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
            print(f"    Avg. Inference Time: {metrics.get('inference_time_avg', 0):.2f}s")
            print(f"    Memory Usage: {memory_usage:.1f} MB")
            print(f"    Reasoning Rate: {metrics.get('reasoning_rate', 0) * 100:.1f}%")
            print(f"    Avg. Reasoning Length: {metrics.get('reasoning_length_avg', 0):.0f} words")
            print(f"    Total Time: {total_time:.1f}s")
    
    # Print visualization location
    print(f"\nVisualizations saved to {os.path.join(config.output_dir, 'visualizations')}")


if __name__ == "__main__":
    main()
