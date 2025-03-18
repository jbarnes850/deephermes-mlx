"""
Fine-tuning module for DeepHermes MLX models.

This module provides functionality for fine-tuning DeepHermes models
using LoRA (Low-Rank Adaptation) on Apple Silicon hardware.
"""

from .lora import apply_lora_to_model, save_lora_weights, load_lora_weights, fuse_lora_weights
from .trainer import train_model, TrainingConfig, prepare_batch, iterate_batches, compute_loss

__all__ = [
    "apply_lora_to_model",
    "save_lora_weights",
    "load_lora_weights",
    "fuse_lora_weights",
    "train_model",
    "TrainingConfig",
    "prepare_batch",
    "iterate_batches",
    "compute_loss",
]
