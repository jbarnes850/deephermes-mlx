#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) implementation for MLX.

This module provides a simplified implementation of LoRA for fine-tuning
MLX models with minimal memory overhead.
"""
import os
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).
    
    This implements a linear layer with a low-rank adaptation that allows
    efficient fine-tuning by only updating the low-rank matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Initialize a LoRALinear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Rank of the low-rank matrices
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA
            bias: Whether to use bias
        """
        super().__init__()
        
        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA components
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize LoRA weights
        # Down projection with normal distribution
        self.lora_A.weight = mx.random.normal(scale=1/r, shape=(r, in_features))
        # Up projection initialized to zero
        self.lora_B.weight = mx.zeros((out_features, r))
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Store dimensions for reference
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Base linear output
        base_output = self.linear(x)
        
        # LoRA path
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
        
        # Combine outputs with scaling
        return base_output + lora_output * self.scaling
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, lora_alpha: float = 16, lora_dropout: float = 0.0) -> "LoRALinear":
        """
        Create a LoRALinear layer from a base linear layer.
        
        Args:
            linear: Base linear layer
            r: Rank of the low-rank matrices
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA
            
        Returns:
            LoRALinear layer
        """
        # Create new LoRA layer
        lora_layer = cls(
            in_features=linear.weight.shape[1],
            out_features=linear.weight.shape[0],
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=hasattr(linear, 'bias') and linear.bias is not None,
        )
        
        # Copy weights from base layer
        lora_layer.linear.weight = linear.weight
        if hasattr(linear, 'bias') and linear.bias is not None:
            lora_layer.linear.bias = linear.bias
        
        return lora_layer


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to a model.
    
    This function replaces linear layers in the model with LoRA layers.
    
    Args:
        model: Model to apply LoRA to
        rank: Rank of the low-rank matrices
        alpha: Alpha parameter for LoRA scaling
        dropout: Dropout probability for LoRA
        target_modules: List of module names to apply LoRA to. If None, applies to all Linear layers.
        
    Returns:
        Model with LoRA applied
    """
    # Default target modules for transformer models
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Count how many layers we've modified
    lora_layer_count = 0
    
    # For MLX models, we need to look at the model's attributes directly
    # First, check if the model has a model attribute (common in MLX-LM models)
    if hasattr(model, "model"):
        model_to_modify = model.model
    else:
        model_to_modify = model
    
    # Check if the model has transformer layers
    if hasattr(model_to_modify, "layers"):
        layers = model_to_modify.layers
        
        # Iterate through each transformer layer
        for i, layer in enumerate(layers):
            # Check for attention modules
            if hasattr(layer, "attention"):
                attention = layer.attention
                
                # Apply LoRA to attention projections
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attention, name) and name in target_modules:
                        proj = getattr(attention, name)
                        if isinstance(proj, nn.Linear):
                            setattr(attention, name, LoRALinear.from_linear(
                                proj, r=rank, lora_alpha=alpha, lora_dropout=dropout
                            ))
                            lora_layer_count += 1
            
            # Check for MLP modules
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # Apply LoRA to MLP projections
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(mlp, name) and name in target_modules:
                        proj = getattr(mlp, name)
                        if isinstance(proj, nn.Linear):
                            setattr(mlp, name, LoRALinear.from_linear(
                                proj, r=rank, lora_alpha=alpha, lora_dropout=dropout
                            ))
                            lora_layer_count += 1
    
    print(f"Applied LoRA to {lora_layer_count} layers with rank {rank}")
    
    return model


def get_lora_params(model: nn.Module) -> Dict[str, mx.array]:
    """
    Extract LoRA parameters from a model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Dictionary of LoRA parameters
    """
    lora_params = {}
    
    # Iterate through all modules
    for name, module in model.named_modules():
        # Check if this is a LoRA layer
        if isinstance(module, LoRALinear):
            # Extract LoRA A and B matrices
            lora_params[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_params[f"{name}.lora_B.weight"] = module.lora_B.weight
    
    return lora_params


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save LoRA weights to a file.
    
    Args:
        model: Model with LoRA layers
        path: Path to save the weights to
    """
    # Extract LoRA parameters
    lora_params = get_lora_params(model)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save parameters
    mx.save(path, lora_params)
    print(f"Saved {len(lora_params)} LoRA parameters to {path}")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load LoRA weights from a file.
    
    Args:
        model: Model with LoRA layers
        path: Path to load the weights from
        
    Returns:
        Model with loaded LoRA weights
    """
    # Load parameters
    lora_params = mx.load(path)
    
    # Iterate through all modules
    for name, module in model.named_modules():
        # Check if this is a LoRA layer
        if isinstance(module, LoRALinear):
            # Load LoRA A and B matrices if they exist
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            
            if a_key in lora_params:
                module.lora_A.weight = lora_params[a_key]
            
            if b_key in lora_params:
                module.lora_B.weight = lora_params[b_key]
    
    print(f"Loaded {len(lora_params)} LoRA parameters from {path}")
    return model


def fuse_lora_weights(model: nn.Module) -> nn.Module:
    """
    Fuse LoRA weights into the base model weights.
    
    This function merges the low-rank adaptation weights with the base model
    weights, effectively applying the fine-tuning permanently.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Model with fused weights (LoRA weights merged into base weights)
    """
    # Counter for fused layers
    fused_count = 0
    
    # For MLX models, we need to look at the model's attributes directly
    # First, check if the model has a model attribute (common in MLX-LM models)
    if hasattr(model, "model"):
        model_to_modify = model.model
    else:
        model_to_modify = model
    
    # Check if the model has transformer layers
    if hasattr(model_to_modify, "layers"):
        layers = model_to_modify.layers
        
        # Iterate through each transformer layer
        for i, layer in enumerate(layers):
            # Check for attention modules
            if hasattr(layer, "attention"):
                attention = layer.attention
                
                # Fuse LoRA in attention projections
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attention, name):
                        proj = getattr(attention, name)
                        if isinstance(proj, LoRALinear):
                            # Compute the LoRA contribution: B × A × scaling
                            lora_contribution = mx.matmul(
                                proj.lora_B.weight, 
                                proj.lora_A.weight
                            ) * proj.scaling
                            
                            # Add the contribution to the base weights
                            proj.linear.weight = proj.linear.weight + lora_contribution
                            
                            # Reset LoRA weights to identity mapping
                            proj.lora_A.weight = mx.zeros_like(proj.lora_A.weight)
                            proj.lora_B.weight = mx.zeros_like(proj.lora_B.weight)
                            
                            fused_count += 1
            
            # Check for MLP modules
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # Fuse LoRA in MLP projections
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(mlp, name):
                        proj = getattr(mlp, name)
                        if isinstance(proj, LoRALinear):
                            # Compute the LoRA contribution: B × A × scaling
                            lora_contribution = mx.matmul(
                                proj.lora_B.weight, 
                                proj.lora_A.weight
                            ) * proj.scaling
                            
                            # Add the contribution to the base weights
                            proj.linear.weight = proj.linear.weight + lora_contribution
                            
                            # Reset LoRA weights to identity mapping
                            proj.lora_A.weight = mx.zeros_like(proj.lora_A.weight)
                            proj.lora_B.weight = mx.zeros_like(proj.lora_B.weight)
                            
                            fused_count += 1
    
    print(f"Fused LoRA weights for {fused_count} layers")
    
    return model
