"""
Dataset loader for benchmarking DeepHermes models.

This module provides utilities for loading and processing datasets
for benchmarking reasoning capabilities.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import os
import json
import random
import csv
import glob
import requests
from tqdm import tqdm
import fnmatch


@dataclass
class Dataset:
    """Dataset for benchmarking."""
    name: str
    samples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def download_file(url: str, local_path: str) -> None:
    """Download a file from a URL to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_path)) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def load_mmlu_dataset(subset: str = "stem", num_samples: Optional[int] = None) -> Dataset:
    """
    Load the MMLU (Massive Multitask Language Understanding) dataset.
    
    Args:
        subset: The subset of MMLU to load (default: "stem")
        num_samples: Number of samples to load (default: all)
    
    Returns:
        Dataset object containing the loaded samples
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define MMLU subsets and their file patterns
    mmlu_subsets = {
        "stem": ["*mathematics*_test.csv", "*physics*_test.csv", "*chemistry*_test.csv", 
                 "*computer*_test.csv", "*engineering*_test.csv", "astronomy_test.csv"],
        "humanities": ["philosophy_test.csv", "*history*_test.csv", "world_religions_test.csv",
                      "jurisprudence_test.csv", "high_school_european_history_test.csv"],
        "social_sciences": ["*economics*_test.csv", "sociology_test.csv", "high_school_psychology_test.csv",
                           "human_sexuality_test.csv", "security_studies_test.csv"],
        "other": ["virology_test.csv", "global_facts_test.csv", "miscellaneous_test.csv"]
    }
    
    if subset not in mmlu_subsets:
        raise ValueError(f"Invalid MMLU subset: {subset}. Available subsets: {list(mmlu_subsets.keys())}")
    
    # Path to the MMLU test data directory
    mmlu_dir = os.path.join(cache_dir, "data", "test")
    
    # Check if MMLU data exists
    if not os.path.exists(mmlu_dir):
        raise FileNotFoundError(f"MMLU dataset not found at {mmlu_dir}. Please download it first.")
    
    # Load the dataset files based on the subset patterns
    samples = []
    patterns = mmlu_subsets[subset]
    
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(mmlu_dir, pattern)):
            print(f"Loading MMLU file: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 6:  # Question, A, B, C, D, Answer
                        question = row[0]
                        options = {
                            'A': row[1],
                            'B': row[2],
                            'C': row[3],
                            'D': row[4]
                        }
                        answer = row[5]
                        
                        # Format the input as a multiple-choice question
                        input_text = f"{question}\n\nOptions:\nA. {options['A']}\nB. {options['B']}\nC. {options['C']}\nD. {options['D']}\n\nPlease select the correct answer."
                        
                        samples.append({
                            "input": input_text,
                            "expected": answer,
                            "options": options,
                            "source_file": os.path.basename(file_path)
                        })
    
    # Limit the number of samples if specified
    if num_samples is not None and num_samples < len(samples):
        samples = random.sample(samples, num_samples)
    
    return Dataset(
        name=f"mmlu_{subset}",
        samples=samples,
        metadata={
            "subset": subset,
            "num_samples": len(samples),
            "source_files": [os.path.basename(f) for f in glob.glob(os.path.join(mmlu_dir, "*_test.csv")) 
                            if any(fnmatch.fnmatch(os.path.basename(f), p) for p in patterns)]
        }
    )


def load_gsm8k_dataset(num_samples: Optional[int] = None) -> Dataset:
    """
    Load the GSM8K (Grade School Math 8K) dataset.
    
    Args:
        num_samples: Number of samples to load (default: all)
    
    Returns:
        Dataset object containing the loaded samples
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached")
    os.makedirs(cache_dir, exist_ok=True)
    
    # GSM8K test set URL
    gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    local_path = os.path.join(cache_dir, "gsm8k_test.jsonl")
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(local_path):
        print("Downloading GSM8K dataset...")
        download_file(gsm8k_url, local_path)
    
    # Load the dataset
    samples = []
    with open(local_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data.get("question", "")
            answer = data.get("answer", "")
            
            # Extract just the final answer from the answer field
            final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
            
            samples.append({
                "input": question,
                "expected": final_answer,
                "full_answer": answer
            })
    
    # Limit the number of samples if specified
    if num_samples is not None and num_samples < len(samples):
        samples = random.sample(samples, num_samples)
    
    return Dataset(
        name="gsm8k",
        samples=samples,
        metadata={
            "num_samples": len(samples)
        }
    )


def load_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional arguments to pass to the dataset loader
    
    Returns:
        Dataset object containing the loaded samples
    """
    if dataset_name.lower() == "mmlu":
        subset = kwargs.get("mmlu_subset", "stem")
        return load_mmlu_dataset(subset=subset, num_samples=kwargs.get("num_samples"))
    elif dataset_name.lower() == "gsm8k":
        return load_gsm8k_dataset(num_samples=kwargs.get("num_samples"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
