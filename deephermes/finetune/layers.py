"""
LoRA layer implementations for DeepHermes MLX models.

This module provides the LoRA layer implementations for fine-tuning DeepHermes models.
"""
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA implementation for linear layers.
    
    This implementation follows the paper "LoRA: Low-Rank Adaptation of Large Language Models"
    (https://arxiv.org/abs/2106.09685).
    """
    
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8, scale: float = 20.0):
        """
        Create a LoRALinear layer from an existing linear layer.
        
        Args:
            linear: The linear layer to convert
            rank: Rank of the low-rank matrices
            scale: Scaling factor
            
        Returns:
            A LoRALinear layer
        """
        # Get input and output dimensions from the linear layer
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
            
        # Create a LoRALinear layer
        lora_lin = LoRALinear(input_dims, output_dims, rank, bias="bias" in linear, scale=scale)
        lora_lin.linear = linear
        return lora_lin
    
    def to_linear(self):
        """
        Convert the LoRALinear layer back to a regular linear layer with fused weights.
        
        Returns:
            A linear layer with fused weights
        """
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)
        
        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype
        
        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
            
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)
        
        # Fuse the weights
        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        
        if bias:
            fused_linear.bias = linear.bias
            
        if is_quantized:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )
            
        return fused_linear
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        bias: bool = False,
        scale: float = 20.0,
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            input_dims: Size of each input sample
            output_dims: Size of each output sample
            lora_rank: Rank of the low-rank matrices
            bias: If set to True, the layer will learn an additive bias
            scale: Scaling factor
        """
        super().__init__()
        
        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        
        # Scale for low-rank update
        self.scale = scale
        
        # Low rank lora weights
        scale_init = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale_init,
            high=scale_init,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))
    
    def __call__(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
            
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        
        return y + self.scale * z


def find_linear_layers(model: Any) -> List[Tuple[List[str], nn.Linear]]:
    """
    Find all linear layers in a model.
    
    Args:
        model: The model to search
        
    Returns:
        A list of (path, layer) tuples
    """
    linear_layers = []
    
    def _find_linear_layers_mlx(obj: Any, path: List[str] = []) -> None:
        """Recursively find linear layers in the MLX model."""
        if isinstance(obj, nn.Linear):
            linear_layers.append((path.copy(), obj))
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = path + [key]
                _find_linear_layers_mlx(value, new_path)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                new_path = path + [str(i)]
                _find_linear_layers_mlx(value, new_path)
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    new_path = path + [key]
                    _find_linear_layers_mlx(value, new_path)
    
    # Start recursive search
    _find_linear_layers_mlx(model, [])
    
    # Sort layers by path to ensure deterministic behavior
    linear_layers.sort(key=lambda x: '.'.join(x[0]))
    
    return linear_layers


def linear_to_lora_layers(
    model: Any,
    num_layers: int,
    lora_params: Dict[str, Union[int, float]],
) -> None:
    """
    Convert linear layers in a model to LoRA layers.
    
    Args:
        model: Model to convert
        num_layers: Number of layers to convert
        lora_params: LoRA parameters (r, alpha, dropout)
    """
    # Find all linear layers in the model
    linear_layers = find_linear_layers(model)
    
    # Convert only the specified number of layers
    for i, (path, layer) in enumerate(linear_layers):
        if i >= num_layers:
            break
        
        # Create LoRA layer
        lora_layer = LoRALinear.from_linear(
            layer,
            rank=lora_params.get("r", 8),
            scale=lora_params.get("alpha", 20.0),
        )
        
        # Replace original layer with LoRA layer
        # Navigate to parent object and set attribute
        parent = model
        for j, part in enumerate(path[:-1]):
            if hasattr(parent, part):
                parent = getattr(parent, part)
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, (list, tuple)) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                # If we can't navigate further, skip this layer
                break
        
        # Set the attribute on the parent
        last_part = path[-1]
        if hasattr(parent, last_part):
            setattr(parent, last_part, lora_layer)
        elif isinstance(parent, dict) and last_part in parent:
            parent[last_part] = lora_layer
        elif isinstance(parent, list) and last_part.isdigit() and int(last_part) < len(parent):
            parent[int(last_part)] = lora_layer
