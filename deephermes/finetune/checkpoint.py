"""
Checkpoint module for DeepHermes MLX fine-tuning.

This module provides functionality for saving and loading model checkpoints
during fine-tuning.
"""
from typing import Dict, Any, Optional, Union
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    metrics: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
    path: Union[str, Path] = "./checkpoint",
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (optional)
        metrics: Metrics to save (optional)
        step: Current step (optional)
        path: Path to save the checkpoint
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    mx.save(path / "weights.npz", model.trainable_parameters())
    
    # Save optimizer state if provided
    if optimizer is not None:
        mx.save(path / "optimizer.npz", optimizer.state)
    
    # Save metadata
    metadata = {
        "timestamp": time.time(),
        "step": step,
    }
    
    # Add metrics if provided
    if metrics is not None:
        # Convert any mx.array values to Python scalars
        processed_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                processed_metrics[k] = {
                    inner_k: float(inner_v) if isinstance(inner_v, mx.array) else inner_v
                    for inner_k, inner_v in v.items()
                }
            elif isinstance(v, mx.array):
                processed_metrics[k] = float(v)
            else:
                processed_metrics[k] = v
        
        metadata["metrics"] = processed_metrics
    
    # Save metadata
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    path: Union[str, Path] = "./checkpoint",
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        path: Path to load the checkpoint from
        
    Returns:
        Dictionary with checkpoint metadata
    """
    path = Path(path)
    
    # Load model weights
    weights = mx.load(path / "weights.npz")
    model.update(weights)
    
    # Load optimizer state if provided
    if optimizer is not None and (path / "optimizer.npz").exists():
        optimizer_state = mx.load(path / "optimizer.npz")
        optimizer.state.update(optimizer_state)
    
    # Load metadata
    metadata = {}
    if (path / "metadata.json").exists():
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
    
    return metadata


def list_checkpoints(
    directory: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    List all checkpoints in a directory.
    
    Args:
        directory: Directory to search for checkpoints
        
    Returns:
        Dictionary mapping checkpoint names to metadata
    """
    directory = Path(directory)
    checkpoints = {}
    
    # Find all subdirectories that contain a weights.npz file
    for path in directory.glob("**/weights.npz"):
        checkpoint_dir = path.parent
        
        # Load metadata if available
        metadata = {}
        if (checkpoint_dir / "metadata.json").exists():
            with open(checkpoint_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
        
        # Add to checkpoints dictionary
        checkpoints[checkpoint_dir.name] = metadata
    
    return checkpoints


def export_model(
    model: nn.Module,
    path: Union[str, Path] = "./exported_model",
    format: str = "mlx",
) -> None:
    """
    Export a model for deployment.
    
    Args:
        model: Model to export
        path: Path to save the exported model
        format: Export format (currently only 'mlx' is supported)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "mlx":
        # Save all parameters (not just trainable ones)
        mx.save(path / "weights.npz", model.parameters())
        
        # Save model metadata
        metadata = {
            "timestamp": time.time(),
            "format": "mlx",
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")
