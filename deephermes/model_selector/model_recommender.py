"""
Model recommendation system for DeepHermes models.

This module provides utilities for recommending the optimal DeepHermes model
and configuration based on detected hardware capabilities.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .hardware_detection import HardwareInfo, get_hardware_info


@dataclass
class ModelConfig:
    """Configuration for a DeepHermes model."""
    model_id: str
    quantization: Optional[str]
    lazy_load: bool
    max_tokens: int
    memory_required_gb: float
    performance_score: int  # Higher is better
    reasoning_quality_score: int  # Higher is better


@dataclass
class ModelRecommendation:
    """Recommendation for a DeepHermes model configuration."""
    model_config: ModelConfig
    confidence: float  # 0.0 to 1.0
    reason: str
    alternatives: List[ModelConfig]


# Model specifications
DEEPHERMES_MODELS = {
    "3B": {
        "model_id": "mlx-community/DeepHermes-3-Llama-3-3B-Preview-bf16",
        "memory_required": {
            "none": 6.0,  # GB, no quantization
            "8bit": 3.0,  # GB, 8-bit quantization
            "4bit": 1.5   # GB, 4-bit quantization
        },
        "performance_score": 90,  # Faster
        "reasoning_quality_score": 70  # Good but not as comprehensive
    },
    "8B": {
        "model_id": "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
        "memory_required": {
            "none": 16.0,  # GB, no quantization
            "8bit": 8.0,   # GB, 8-bit quantization
            "4bit": 4.0    # GB, 4-bit quantization
        },
        "performance_score": 75,  # Good balance
        "reasoning_quality_score": 85  # Very good reasoning
    },
    "24B": {
        "model_id": "mlx-community/DeepHermes-3-Mistral-24B-Preview-bf16",
        "memory_required": {
            "none": 48.0,  # GB, no quantization
            "8bit": 24.0,  # GB, 8-bit quantization
            "4bit": 12.0   # GB, 4-bit quantization
        },
        "performance_score": 40,  # Slower
        "reasoning_quality_score": 95  # Excellent reasoning
    }
}


def get_model_configs() -> List[ModelConfig]:
    """
    Get all possible model configurations.
    
    Returns:
        List of all possible model configurations
    """
    configs = []
    
    for size, specs in DEEPHERMES_MODELS.items():
        for quant, memory in specs["memory_required"].items():
            # Convert "none" to None for the API
            quantization = None if quant == "none" else quant
            
            # Create configurations with and without lazy loading
            for lazy in [True, False]:
                # Skip non-lazy loading for large models that would require too much memory
                if not lazy and memory > 20:
                    continue
                
                configs.append(ModelConfig(
                    model_id=specs["model_id"],
                    quantization=quantization,
                    lazy_load=lazy,
                    max_tokens=1024,  # Default
                    memory_required_gb=memory * (0.7 if lazy else 1.0),  # Lazy loading reduces memory
                    performance_score=specs["performance_score"] * (0.9 if lazy else 1.0),  # Lazy loading slightly reduces performance
                    reasoning_quality_score=specs["reasoning_quality_score"]
                ))
    
    return configs


def recommend_model(hardware_info: Optional[HardwareInfo] = None) -> ModelRecommendation:
    """
    Recommend the optimal DeepHermes model based on hardware capabilities.
    
    Args:
        hardware_info: Hardware information, or None to detect automatically
        
    Returns:
        ModelRecommendation with the optimal model configuration
    """
    if hardware_info is None:
        hardware_info = get_hardware_info()
    
    # Get available memory with a safety margin
    available_memory_gb = hardware_info.memory_gb * 0.8  # 80% of total memory
    
    # Get all possible configurations
    all_configs = get_model_configs()
    
    # Filter configurations that fit in available memory
    viable_configs = [
        config for config in all_configs 
        if config.memory_required_gb <= available_memory_gb
    ]
    
    if not viable_configs:
        # If no configurations fit, recommend the smallest model with 4-bit quantization
        smallest_config = min(all_configs, key=lambda c: c.memory_required_gb)
        return ModelRecommendation(
            model_config=smallest_config,
            confidence=0.3,
            reason="Warning: All models exceed available memory. This configuration may cause system instability.",
            alternatives=[]
        )
    
    # Score configurations based on hardware capabilities
    scored_configs = []
    for config in viable_configs:
        # Base score is a combination of performance and reasoning quality
        base_score = (config.performance_score + config.reasoning_quality_score) / 2
        
        # Adjust score based on memory headroom
        memory_headroom = available_memory_gb - config.memory_required_gb
        memory_score = min(memory_headroom / 2, 10)  # Cap at 10 points
        
        # Adjust score based on Apple Silicon capabilities
        apple_silicon_bonus = 0
        if hardware_info.is_apple_silicon:
            # Larger models benefit more from Neural Engine
            if "24B" in config.model_id and hardware_info.neural_engine_cores and hardware_info.neural_engine_cores >= 16:
                apple_silicon_bonus = 15
            elif "8B" in config.model_id and hardware_info.neural_engine_cores and hardware_info.neural_engine_cores >= 16:
                apple_silicon_bonus = 10
            elif "3B" in config.model_id:
                apple_silicon_bonus = 5
        
        # Calculate final score
        final_score = base_score + memory_score + apple_silicon_bonus
        
        scored_configs.append((config, final_score))
    
    # Sort by score (descending)
    scored_configs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendation and alternatives
    top_config, top_score = scored_configs[0]
    
    # Calculate confidence based on score difference with next best
    confidence = 0.8  # Base confidence
    if len(scored_configs) > 1:
        score_diff = top_score - scored_configs[1][1]
        confidence = min(0.95, 0.8 + (score_diff / 50))  # Adjust based on score difference
    
    # Generate reason for recommendation
    if "24B" in top_config.model_id:
        reason = "The 24B model is recommended for best reasoning quality. "
    elif "8B" in top_config.model_id:
        reason = "The 8B model is recommended for balanced performance and reasoning quality. "
    else:
        reason = "The 3B model is recommended for fastest performance. "
    
    if top_config.quantization:
        reason += f"{top_config.quantization} quantization helps reduce memory usage while maintaining good quality. "
    else:
        reason += "Full precision (no quantization) provides the highest quality output. "
    
    if top_config.lazy_load:
        reason += "Lazy loading is enabled to optimize memory usage."
    
    # Get alternatives (up to 2)
    alternatives = [config for config, _ in scored_configs[1:3]]
    
    return ModelRecommendation(
        model_config=top_config,
        confidence=confidence,
        reason=reason,
        alternatives=alternatives
    )


def print_recommendation(recommendation: ModelRecommendation) -> None:
    """
    Print a model recommendation in a user-friendly format.
    
    Args:
        recommendation: The model recommendation to print
    """
    config = recommendation.model_config
    
    print("\n===== DeepHermes Model Recommendation =====")
    print(f"Recommended Model: {config.model_id.split('/')[-1]}")
    print(f"Quantization: {config.quantization if config.quantization else 'None (full precision)'}")
    print(f"Lazy Loading: {'Enabled' if config.lazy_load else 'Disabled'}")
    print(f"Confidence: {int(recommendation.confidence * 100)}%")
    print(f"Reason: {recommendation.reason}")
    
    if recommendation.alternatives:
        print("\nAlternative Options:")
        for i, alt in enumerate(recommendation.alternatives, 1):
            model_name = alt.model_id.split('/')[-1]
            quant_str = f"with {alt.quantization} quantization" if alt.quantization else "without quantization"
            print(f"  {i}. {model_name} {quant_str}")
    
    # Generate command to use this configuration
    cmd = f"python chat.py --model {config.model_id} "
    if config.quantization:
        cmd += f"--quantize {config.quantization} "
    if config.lazy_load:
        cmd += "--lazy-load "
    cmd += "--reasoning"
    
    print("\nCommand to use this configuration:")
    print(cmd)
    print("=============================================")


if __name__ == "__main__":
    # Example usage
    hardware = get_hardware_info()
    recommendation = recommend_model(hardware)
    print_recommendation(recommendation)
