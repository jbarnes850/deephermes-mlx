"""
Metrics for evaluating DeepHermes model performance.

This module provides metrics for evaluating model performance on reasoning tasks,
including accuracy, reasoning quality, and efficiency metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np
from dataclasses import dataclass


def extract_thinking(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking process and final answer from model output.
    
    Args:
        text: Model output text
        
    Returns:
        Tuple of (thinking_process, final_answer)
    """
    # Extract thinking process within <think> tags
    thinking_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else None
    
    # Extract final answer (everything after the last </think> tag)
    if thinking_match:
        final_answer = text[thinking_match.end():].strip()
    else:
        final_answer = text.strip()
    
    return thinking, final_answer


def calculate_mmlu_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for MMLU dataset.
    
    Args:
        samples: List of sample results
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": 0.0,
        "reasoning_rate": 0.0,
        "reasoning_length_avg": 0.0,
        "reasoning_length_std": 0.0,
        "inference_time_avg": 0.0,
        "inference_time_std": 0.0,
    }
    
    if not samples:
        return metrics
    
    # Calculate accuracy
    correct = 0
    reasoning_count = 0
    reasoning_lengths = []
    inference_times = []
    
    for sample in samples:
        output = sample.get("output", "")
        expected = sample.get("expected", "").strip()
        
        # Extract thinking and final answer
        thinking, final_answer = extract_thinking(output)
        
        # Check for correct answer (looking for A, B, C, or D in the final answer)
        answer_match = re.search(r'\b([A-D])\b', final_answer)
        predicted = answer_match.group(1) if answer_match else ""
        
        if predicted == expected:
            correct += 1
        
        # Track reasoning metrics
        if thinking:
            reasoning_count += 1
            reasoning_lengths.append(len(thinking.split()))
        
        # Track inference time
        inference_times.append(sample.get("time_seconds", 0))
    
    # Calculate metrics
    metrics["accuracy"] = correct / len(samples) if samples else 0
    metrics["reasoning_rate"] = reasoning_count / len(samples) if samples else 0
    
    if reasoning_lengths:
        metrics["reasoning_length_avg"] = np.mean(reasoning_lengths)
        metrics["reasoning_length_std"] = np.std(reasoning_lengths)
    
    if inference_times:
        metrics["inference_time_avg"] = np.mean(inference_times)
        metrics["inference_time_std"] = np.std(inference_times)
    
    return metrics


def calculate_gsm8k_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for GSM8K dataset.
    
    Args:
        samples: List of sample results
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": 0.0,
        "reasoning_rate": 0.0,
        "reasoning_length_avg": 0.0,
        "reasoning_length_std": 0.0,
        "inference_time_avg": 0.0,
        "inference_time_std": 0.0,
    }
    
    if not samples:
        return metrics
    
    # Calculate accuracy
    correct = 0
    reasoning_count = 0
    reasoning_lengths = []
    inference_times = []
    
    for sample in samples:
        output = sample.get("output", "")
        expected = sample.get("expected", "").strip()
        
        # Extract thinking and final answer
        thinking, final_answer = extract_thinking(output)
        
        # Extract numerical answer from the final answer
        # Look for numbers in the final answer
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', final_answer)
        predicted = numbers[-1] if numbers else ""
        
        # Normalize expected answer (remove commas, units, etc.)
        expected_normalized = re.sub(r'[^\d.]', '', expected)
        predicted_normalized = re.sub(r'[^\d.]', '', predicted)
        
        if predicted_normalized == expected_normalized:
            correct += 1
        
        # Track reasoning metrics
        if thinking:
            reasoning_count += 1
            reasoning_lengths.append(len(thinking.split()))
        
        # Track inference time
        inference_times.append(sample.get("time_seconds", 0))
    
    # Calculate metrics
    metrics["accuracy"] = correct / len(samples) if samples else 0
    metrics["reasoning_rate"] = reasoning_count / len(samples) if samples else 0
    
    if reasoning_lengths:
        metrics["reasoning_length_avg"] = np.mean(reasoning_lengths)
        metrics["reasoning_length_std"] = np.std(reasoning_lengths)
    
    if inference_times:
        metrics["inference_time_avg"] = np.mean(inference_times)
        metrics["inference_time_std"] = np.std(inference_times)
    
    return metrics


def calculate_metrics(samples: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """
    Calculate metrics for a dataset.
    
    Args:
        samples: List of sample results
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of metrics
    """
    if dataset_name.startswith("mmlu"):
        return calculate_mmlu_metrics(samples)
    elif dataset_name == "gsm8k":
        return calculate_gsm8k_metrics(samples)
    else:
        # Default metrics calculation
        return {
            "accuracy": 0.0,
            "reasoning_rate": 0.0,
            "reasoning_length_avg": 0.0,
            "reasoning_length_std": 0.0,
            "inference_time_avg": 0.0,
            "inference_time_std": 0.0,
        }
