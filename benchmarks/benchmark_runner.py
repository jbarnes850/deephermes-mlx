"""
Benchmark runner for DeepHermes models.

This module provides utilities for benchmarking DeepHermes models on various datasets
and metrics, with a focus on reasoning capabilities.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import json
import os
import sys
from dataclasses import dataclass, field
import mlx.core as mx
import psutil
from tqdm import tqdm

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deephermes.core.model import load_model
from deephermes.core.inference import run_inference, format_prompt
from deephermes.core.utils import get_reasoning_prompt, get_default_system_prompt
from .datasets.dataset_loader import Dataset, load_dataset
from .metrics.metrics import calculate_metrics
from .visualizations.visualization import generate_visualizations


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_ids: List[str] = field(default_factory=lambda: [
        "mlx-community/DeepHermes-3-Llama-3-3B-Preview-bf16",
        "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
        "mlx-community/DeepHermes-3-Mistral-24B-Preview-bf16"
    ])
    dataset_names: List[str] = field(default_factory=lambda: ["mmlu", "gsm8k"])
    batch_size: int = 1
    max_tokens: int = 1024
    quantization: Optional[str] = None  # "4bit", "8bit", or None
    lazy_load: bool = False
    reasoning: bool = True
    output_dir: str = "benchmarks/results"
    num_samples: Optional[int] = None  # If None, use all samples


class ModelBenchmark:
    """Benchmark runner for DeepHermes models."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark runner with the given configuration."""
        self.config = config
        self.results: Dict[str, Dict[str, Any]] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the benchmark report."""
        import platform
        
        # Get Apple Silicon chip information if available
        chip_info = "Unknown"
        try:
            if platform.system() == "Darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], 
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    chip_info = result.stdout.strip()
        except Exception:
            pass
            
        return {
            "platform": platform.platform(),
            "processor": chip_info,
            "python_version": platform.python_version(),
            "memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "mlx_version": mx.__version__,
        }
    
    def _load_model(self, model_id: str):
        """Load a model for benchmarking."""
        print(f"Loading model: {model_id}")
        model, tokenizer = load_model(
            model_path=model_id, 
            quantize=self.config.quantization,
            lazy_load=self.config.lazy_load
        )
        return model, tokenizer
    
    def _benchmark_model_on_dataset(
        self, 
        model_id: str, 
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Benchmark a single model on a single dataset."""
        model, tokenizer = self._load_model(model_id)
        
        # Track memory usage
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB
        
        results = {
            "model_id": model_id,
            "dataset": dataset.name,
            "samples": [],
            "metrics": {},
            "memory_usage_mb": 0,
            "total_time_seconds": 0,
        }
        
        # Prepare system prompt with reasoning if enabled
        system_prompt = get_default_system_prompt()
        if self.config.reasoning:
            system_prompt += get_reasoning_prompt("deep")
        
        # Run inference on dataset samples
        start_time = time.time()
        
        num_samples = len(dataset.samples)
        if self.config.num_samples is not None:
            num_samples = min(self.config.num_samples, num_samples)
        
        for i, sample in enumerate(tqdm(dataset.samples[:num_samples], desc=f"Evaluating {model_id}")):
            # Create messages for the chat format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": sample["input"]})
            
            # Measure inference time
            sample_start_time = time.time()
            output = run_inference(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=0.0,  # Use 0 for deterministic outputs in benchmarks
                top_p=1.0,
                stream=False  # Don't stream output in benchmarks
            )
            sample_time = time.time() - sample_start_time
            
            # Store sample results
            sample_result = {
                "input": sample["input"],
                "expected": sample["expected"],
                "output": output,
                "time_seconds": sample_time
            }
            results["samples"].append(sample_result)
        
        # Calculate total time
        results["total_time_seconds"] = time.time() - start_time
        
        # Measure memory usage
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB
        results["memory_usage_mb"] = memory_after - memory_before
        
        # Calculate metrics
        results["metrics"] = calculate_metrics(results["samples"], dataset.name)
        
        return results
    
    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all configured models and datasets."""
        system_info = self._get_system_info()
        
        for model_id in self.config.model_ids:
            self.results[model_id] = {
                "model_id": model_id,
                "system_info": system_info,
                "config": vars(self.config),
                "datasets": {}
            }
            
            for dataset_name in self.config.dataset_names:
                dataset = load_dataset(dataset_name)
                
                print(f"\nBenchmarking {model_id} on {dataset_name}...")
                dataset_results = self._benchmark_model_on_dataset(model_id, dataset)
                self.results[model_id]["datasets"][dataset_name] = dataset_results
                
                # Save intermediate results
                self._save_results(model_id)
        
        # Generate visualizations
        generate_visualizations(self.results, os.path.join(self.config.output_dir, "visualizations"))
        
        return self.results
    
    def _save_results(self, model_id: str) -> None:
        """Save benchmark results to disk."""
        output_path = os.path.join(self.config.output_dir, f"{model_id.split('/')[-1]}.json")
        with open(output_path, "w") as f:
            json.dump(self.results[model_id], f, indent=2)


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Run benchmarks with the given configuration."""
    if config is None:
        config = BenchmarkConfig()
    
    benchmark = ModelBenchmark(config)
    return benchmark.run_benchmarks()


if __name__ == "__main__":
    # Example usage
    config = BenchmarkConfig(
        model_ids=["mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16"],
        dataset_names=["mmlu"],
        num_samples=10  # Limit samples for testing
    )
    run_benchmark(config)
