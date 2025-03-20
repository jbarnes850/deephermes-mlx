"""
Configuration manager for the Adaptive ML Workflow.

This module provides a central configuration system that intelligently
configures every aspect of the ML pipeline based on hardware capabilities.
"""

from typing import Dict, Any, Optional, List
import json
import os
from .hardware_profiles import AppleSiliconProfile, detect_hardware
from .workflow_templates import get_workflow_template, apply_workflow_template, WorkflowTemplate
from ..model_selector.model_recommender import ModelConfig, ModelRecommendation, recommend_model
from ..model_selector.hardware_detection import get_hardware_info, HardwareInfo

class AdaptiveWorkflowConfig:
    """
    Central configuration manager for the adaptive ML workflow.
    
    This class manages configurations for all components of the ML pipeline
    based on detected hardware capabilities.
    """
    
    def __init__(self, 
                 hardware_profile: Optional[AppleSiliconProfile] = None,
                 workflow_type: str = "general",
                 prioritize_speed: bool = False,
                 prioritize_quality: bool = False,
                 max_memory_usage_pct: float = 80.0):
        """
        Initialize the configuration manager.
        
        Args:
            hardware_profile: Hardware profile, or None to detect automatically
            workflow_type: Type of workflow (general, content_creation, coding, etc.)
            prioritize_speed: Whether to prioritize speed over quality
            prioritize_quality: Whether to prioritize quality over speed
            max_memory_usage_pct: Maximum percentage of memory to use
        """
        self.hardware_profile = hardware_profile or detect_hardware()
        self.workflow_type = workflow_type
        self.prioritize_speed = prioritize_speed
        self.prioritize_quality = prioritize_quality
        self.max_memory_usage_pct = max_memory_usage_pct
        
        # Get the workflow template
        self.workflow_template = get_workflow_template(workflow_type)
        
        # Generate configurations for all components
        self.model_config = self._generate_model_config()
        self.fine_tuning_config = self._generate_fine_tuning_config()
        self.serving_config = self._generate_serving_config()
        self.integration_config = self._generate_integration_config()
        
        # Apply workflow template adjustments
        self._apply_workflow_template()
        
    def _generate_model_config(self) -> Dict[str, Any]:
        """Generate model configuration based on hardware profile."""
        # Create a HardwareInfo object to use with the existing model_selector
        hardware_info = HardwareInfo(
            device_name=f"{self.hardware_profile.chip_family} {self.hardware_profile.chip_variant}",
            chip_type=f"{self.hardware_profile.chip_family} {self.hardware_profile.chip_variant}",
            memory_gb=float(self.hardware_profile.memory_gb),
            cpu_cores=self.hardware_profile.cpu_cores,
            gpu_cores=self.hardware_profile.gpu_cores,
            neural_engine_cores=self.hardware_profile.neural_engine_cores,
            is_apple_silicon=self.hardware_profile.chip_family in ["M1", "M2", "M3", "M4"]
        )
        
        # Use the existing model_selector to get a recommendation
        recommendation = recommend_model(hardware_info)
        model_config = recommendation.model_config
        
        # Adjust based on priorities if needed
        if self.prioritize_speed and not model_config.quantization:
            # Find a similar model with quantization for faster inference
            for alt in recommendation.alternatives:
                if alt.quantization and alt.model_id == model_config.model_id:
                    model_config = alt
                    break
        elif self.prioritize_quality and model_config.quantization:
            # Find a similar model without quantization for better quality
            for alt in recommendation.alternatives:
                if not alt.quantization and alt.model_id == model_config.model_id:
                    model_config = alt
                    break
        
        # Convert to a dictionary for easier serialization
        return {
            "model_id": model_config.model_id,
            "quantization": model_config.quantization,
            "lazy_load": model_config.lazy_load,
            "max_tokens": model_config.max_tokens,
            "memory_required_gb": model_config.memory_required_gb,
            "recommendation_reason": recommendation.reason,
            "recommendation_confidence": recommendation.confidence
        }
    
    def _generate_fine_tuning_config(self) -> Dict[str, Any]:
        """Generate fine-tuning configuration based on hardware profile."""
        # Determine optimal batch size based on memory
        if self.hardware_profile.memory_gb >= 64:
            batch_size = 4
        elif self.hardware_profile.memory_gb >= 32:
            batch_size = 2
        else:
            batch_size = 1
        
        # Determine LoRA rank based on model size and available memory
        if "24B" in self.model_config["model_id"] and self.hardware_profile.memory_gb >= 64:
            lora_rank = 16
        elif "8B" in self.model_config["model_id"] and self.hardware_profile.memory_gb >= 32:
            lora_rank = 12
        else:
            lora_rank = 8
        
        # Adjust based on priorities
        if self.prioritize_speed:
            lora_rank = max(4, lora_rank - 4)  # Reduce rank for speed
        elif self.prioritize_quality:
            lora_rank = min(32, lora_rank + 4)  # Increase rank for quality
            
        return {
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "learning_rate": 1e-5,
            "max_seq_len": min(2048, 512 * batch_size),
            "epochs": 3,
            "warmup_steps": 100,
            "save_steps": 200,
            "eval_steps": 100,
            "gradient_accumulation_steps": max(1, 4 // batch_size),
            "use_8bit_optimizer": self.hardware_profile.memory_gb < 32,
        }
    
    def _generate_serving_config(self) -> Dict[str, Any]:
        """Generate serving configuration based on hardware profile."""
        # Determine optimal serving parameters based on hardware
        max_concurrent = max(1, self.hardware_profile.cpu_cores // 2)
        
        # Adjust based on priorities
        if self.prioritize_speed:
            max_concurrent = max(1, max_concurrent + 1)
        elif self.prioritize_quality:
            max_concurrent = max(1, max_concurrent - 1)
            
        return {
            "host": "127.0.0.1",
            "port": 8080,
            "max_concurrent_requests": max_concurrent,
            "streaming": True,
            "max_tokens_per_request": 4096,
            "context_window": 8192 if self.hardware_profile.memory_gb >= 32 else 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "system_prompt": self.workflow_template.system_prompt,
            "reasoning_enabled": self.workflow_template.reasoning_enabled,
            "reasoning_depth": self.workflow_template.reasoning_depth,
        }
    
    def _generate_integration_config(self) -> Dict[str, Any]:
        """Generate integration configuration based on hardware profile."""
        # Determine optimal integration parameters
        cache_size = max(1, self.hardware_profile.memory_gb // 8)
        
        # Adjust based on priorities
        if self.prioritize_speed:
            cache_size = max(2, cache_size * 2)  # Larger cache for speed
        elif self.prioritize_quality:
            cache_size = max(1, cache_size // 2)  # Smaller cache to save memory for quality
            
        return {
            "cache_limit_gb": cache_size,
            "streaming": True,
            "timeout_seconds": 60,
            "retry_attempts": 3,
            "chunk_size": 512,
            "use_memory_efficient_attention": self.hardware_profile.memory_gb < 32,
        }
    
    def _apply_workflow_template(self) -> None:
        """Apply the workflow template to the configuration."""
        # Apply model config adjustments
        for key, value in self.workflow_template.model_config_adjustments.items():
            self.model_config[key] = value
        
        # Apply fine-tuning config adjustments
        for key, value in self.workflow_template.fine_tuning_config_adjustments.items():
            self.fine_tuning_config[key] = value
        
        # Apply serving config adjustments
        for key, value in self.workflow_template.serving_config_adjustments.items():
            self.serving_config[key] = value
        
        # Apply integration config adjustments
        for key, value in self.workflow_template.integration_config_adjustments.items():
            self.integration_config[key] = value
        
    def get_full_config(self) -> Dict[str, Any]:
        """Get the full configuration for all components."""
        return {
            "hardware_profile": {
                "chip_family": self.hardware_profile.chip_family,
                "chip_variant": self.hardware_profile.chip_variant,
                "cpu_cores": self.hardware_profile.cpu_cores,
                "performance_cores": self.hardware_profile.performance_cores,
                "efficiency_cores": self.hardware_profile.efficiency_cores,
                "gpu_cores": self.hardware_profile.gpu_cores,
                "neural_engine_cores": self.hardware_profile.neural_engine_cores,
                "memory_gb": self.hardware_profile.memory_gb,
                "memory_bandwidth_gbps": self.hardware_profile.memory_bandwidth_gbps,
                "supports_ray_tracing": self.hardware_profile.supports_ray_tracing,
                "total_compute_power": self.hardware_profile.total_compute_power,
            },
            "model_config": self.model_config,
            "fine_tuning_config": self.fine_tuning_config,
            "serving_config": self.serving_config,
            "integration_config": self.integration_config,
            "workflow_type": self.workflow_type,
            "performance_targets": {
                "prioritize_speed": self.prioritize_speed,
                "prioritize_quality": self.prioritize_quality,
                "max_memory_usage_pct": self.max_memory_usage_pct,
            },
            "workflow_template": {
                "name": self.workflow_template.name,
                "description": self.workflow_template.description,
                "system_prompt": self.workflow_template.system_prompt,
                "reasoning_enabled": self.workflow_template.reasoning_enabled,
                "reasoning_depth": self.workflow_template.reasoning_depth,
            }
        }
    
    def save_config(self, file_path: str) -> None:
        """Save the configuration to a file."""
        with open(file_path, 'w') as f:
            json.dump(self.get_full_config(), f, indent=2)
    
    @classmethod
    def load_config(cls, file_path: str) -> 'AdaptiveWorkflowConfig':
        """Load a configuration from a file."""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
            
        # Reconstruct the hardware profile
        hw_data = config_data["hardware_profile"]
        hardware_profile = AppleSiliconProfile(
            chip_family=hw_data["chip_family"],
            chip_variant=hw_data["chip_variant"],
            cpu_cores=hw_data["cpu_cores"],
            performance_cores=hw_data["performance_cores"],
            efficiency_cores=hw_data["efficiency_cores"],
            gpu_cores=hw_data["gpu_cores"],
            neural_engine_cores=hw_data["neural_engine_cores"],
            memory_gb=hw_data["memory_gb"],
            memory_bandwidth_gbps=hw_data["memory_bandwidth_gbps"],
            supports_ray_tracing=hw_data["supports_ray_tracing"]
        )
        
        # Create a new instance with the loaded configuration
        instance = cls(
            hardware_profile=hardware_profile,
            workflow_type=config_data["workflow_type"],
            prioritize_speed=config_data["performance_targets"]["prioritize_speed"],
            prioritize_quality=config_data["performance_targets"]["prioritize_quality"],
            max_memory_usage_pct=config_data["performance_targets"]["max_memory_usage_pct"],
        )
        
        # Override the generated configurations with the loaded ones
        instance.model_config = config_data["model_config"]
        instance.fine_tuning_config = config_data["fine_tuning_config"]
        instance.serving_config = config_data["serving_config"]
        instance.integration_config = config_data["integration_config"]
        
        return instance