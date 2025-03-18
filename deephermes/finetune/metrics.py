"""
Metrics module for DeepHermes MLX fine-tuning.

This module provides functionality for tracking and computing metrics
during model training and evaluation.
"""
from typing import Dict, Any, List, Optional
from collections import defaultdict


class MetricsTracker:
    """Class for tracking metrics during training and evaluation."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        self.steps = []
        
    def update_train_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update training metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (optional)
        """
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
        
        if step is not None:
            self.steps.append(step)
    
    def update_eval_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (optional)
        """
        for key, value in metrics.items():
            self.eval_metrics[key].append(value)
    
    def get_train_metrics(self) -> Dict[str, float]:
        """
        Get the latest training metrics.
        
        Returns:
            Dictionary of latest training metrics
        """
        return {key: values[-1] if values else 0.0 for key, values in self.train_metrics.items()}
    
    def get_eval_metrics(self) -> Dict[str, float]:
        """
        Get the latest evaluation metrics.
        
        Returns:
            Dictionary of latest evaluation metrics
        """
        return {key: values[-1] if values else 0.0 for key, values in self.eval_metrics.items()}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "train": {key: values for key, values in self.train_metrics.items()},
            "eval": {key: values for key, values in self.eval_metrics.items()},
            "steps": self.steps,
        }
    
    def get_best_eval_metrics(self) -> Dict[str, float]:
        """
        Get the best evaluation metrics.
        
        Returns:
            Dictionary of best evaluation metrics
        """
        best_metrics = {}
        
        # For loss metrics, lower is better
        for key in ["eval_loss", "eval_perplexity"]:
            if key in self.eval_metrics and self.eval_metrics[key]:
                best_metrics[key] = min(self.eval_metrics[key])
        
        # For accuracy metrics, higher is better
        for key in ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]:
            if key in self.eval_metrics and self.eval_metrics[key]:
                best_metrics[key] = max(self.eval_metrics[key])
        
        return best_metrics


def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with metrics
    """
    # This is a placeholder for more sophisticated metrics
    # In a real implementation, you would use libraries like nltk, rouge, or sacrebleu
    
    # Simple exact match accuracy
    correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return {
        "accuracy": accuracy,
    }
