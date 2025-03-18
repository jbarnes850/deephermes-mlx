"""
Core module for DeepHermes MLX.

This module provides core functionality for model loading and inference.
"""

from deephermes.core.model import (
    load_model,
    load_model_from_dict,
    check_model_compatibility,
    get_verified_compatible_models,
    suggest_alternative_models
)
from deephermes.core.inference import run_inference

__all__ = [
    "load_model",
    "load_model_from_dict",
    "check_model_compatibility",
    "get_verified_compatible_models",
    "suggest_alternative_models",
    "run_inference"
]
