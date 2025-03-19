"""
Hardware profiles for the Adaptive ML Workflow.

This module defines hardware profiles for different Apple Silicon variants
and provides utilities for detecting hardware capabilities.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os
import sys
from ..model_selector.hardware_detection import get_hardware_info, HardwareInfo

@dataclass
class AppleSiliconProfile:
    """Profile for Apple Silicon hardware capabilities."""
    chip_family: str  # M1, M2, M3, M4
    chip_variant: str  # Base, Pro, Max, Ultra
    cpu_cores: int
    performance_cores: int
    efficiency_cores: int
    gpu_cores: int
    neural_engine_cores: int
    memory_gb: int
    memory_bandwidth_gbps: float
    supports_ray_tracing: bool = False
    
    @property
    def total_compute_power(self) -> float:
        """Calculate a normalized compute power score."""
        # Algorithm to estimate total compute capability
        return (self.performance_cores * 2.0 + 
                self.efficiency_cores * 0.5 + 
                self.gpu_cores * 0.8 + 
                self.neural_engine_cores * 0.3)
    
    def supports_model_size(self, model_size_billions: int) -> bool:
        """Determine if this hardware can efficiently run a model of given size."""
        # Logic to determine if hardware can handle model size
        if model_size_billions <= 3:
            return True
        elif model_size_billions <= 8:
            return self.memory_gb >= 16
        elif model_size_billions <= 24:
            return self.memory_gb >= 32 and (
                self.chip_variant in ["Max", "Ultra"] or 
                (self.chip_variant == "Pro" and self.chip_family >= "M3")
            )
        else:
            return self.memory_gb >= 64 and self.chip_variant in ["Max", "Ultra"]

# Define profiles for all current Apple Silicon variants
HARDWARE_PROFILES = {
    # M4 Series
    "M4": AppleSiliconProfile(
        chip_family="M4",
        chip_variant="Base",
        cpu_cores=10,
        performance_cores=4,
        efficiency_cores=6,
        gpu_cores=10,
        neural_engine_cores=16,
        memory_gb=16,
        memory_bandwidth_gbps=100,
        supports_ray_tracing=True
    ),
    "M4_Pro": AppleSiliconProfile(
        chip_family="M4",
        chip_variant="Pro",
        cpu_cores=12,
        performance_cores=8,
        efficiency_cores=4,
        gpu_cores=20,
        neural_engine_cores=16,
        memory_gb=32,
        memory_bandwidth_gbps=200,
        supports_ray_tracing=True
    ),
    "M4_Max": AppleSiliconProfile(
        chip_family="M4",
        chip_variant="Max",
        cpu_cores=16,
        performance_cores=12,
        efficiency_cores=4,
        gpu_cores=40,
        neural_engine_cores=16,
        memory_gb=128,
        memory_bandwidth_gbps=400,
        supports_ray_tracing=True
    ),
    
    # M3 Series (including Ultra)
    "M3_Ultra": AppleSiliconProfile(
        chip_family="M3",
        chip_variant="Ultra",
        cpu_cores=32,
        performance_cores=24,
        efficiency_cores=8,
        gpu_cores=80,
        neural_engine_cores=32,
        memory_gb=512,
        memory_bandwidth_gbps=819,
        supports_ray_tracing=True
    ),
    "M3_Max": AppleSiliconProfile(
        chip_family="M3",
        chip_variant="Max",
        cpu_cores=16,
        performance_cores=12,
        efficiency_cores=4,
        gpu_cores=40,
        neural_engine_cores=16,
        memory_gb=128,
        memory_bandwidth_gbps=400,
        supports_ray_tracing=True
    ),
    "M3_Pro": AppleSiliconProfile(
        chip_family="M3",
        chip_variant="Pro",
        cpu_cores=12,
        performance_cores=6,
        efficiency_cores=6,
        gpu_cores=18,
        neural_engine_cores=16,
        memory_gb=36,
        memory_bandwidth_gbps=150,
        supports_ray_tracing=True
    ),
    "M3": AppleSiliconProfile(
        chip_family="M3",
        chip_variant="Base",
        cpu_cores=8,
        performance_cores=4,
        efficiency_cores=4,
        gpu_cores=10,
        neural_engine_cores=16,
        memory_gb=24,
        memory_bandwidth_gbps=100,
        supports_ray_tracing=True
    ),
}

def map_hardware_info_to_profile(hardware_info: HardwareInfo) -> AppleSiliconProfile:
    """
    Map hardware info from the model_selector to an AppleSiliconProfile.
    
    Args:
        hardware_info: HardwareInfo object from model_selector
        
    Returns:
        Matching AppleSiliconProfile or a custom profile based on detection
    """
    if not hardware_info.is_apple_silicon:
        # Create a generic profile for non-Apple Silicon
        return AppleSiliconProfile(
            chip_family="Generic",
            chip_variant="",
            cpu_cores=hardware_info.cpu_cores,
            performance_cores=hardware_info.cpu_cores // 2,
            efficiency_cores=hardware_info.cpu_cores // 2,
            gpu_cores=0,
            neural_engine_cores=0,
            memory_gb=int(hardware_info.memory_gb),
            memory_bandwidth_gbps=50,
            supports_ray_tracing=False
        )
    
    # Parse chip type to extract family and variant
    chip_parts = hardware_info.chip_type.split()
    chip_family = chip_parts[0] if chip_parts else "Unknown"
    chip_variant = chip_parts[1] if len(chip_parts) > 1 else "Base"
    
    # Try to find a matching profile
    profile_key = f"{chip_family}_{chip_variant}" if chip_variant != "Base" else chip_family
    if profile_key in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[profile_key]
        
        # Update memory if it differs from the profile
        if abs(profile.memory_gb - hardware_info.memory_gb) > 2:
            # Create a copy with the correct memory
            return AppleSiliconProfile(
                chip_family=profile.chip_family,
                chip_variant=profile.chip_variant,
                cpu_cores=profile.cpu_cores,
                performance_cores=profile.performance_cores,
                efficiency_cores=profile.efficiency_cores,
                gpu_cores=profile.gpu_cores,
                neural_engine_cores=profile.neural_engine_cores,
                memory_gb=int(hardware_info.memory_gb),
                memory_bandwidth_gbps=profile.memory_bandwidth_gbps,
                supports_ray_tracing=profile.supports_ray_tracing
            )
        
        return profile
    
    # If no matching profile, create a custom one
    return AppleSiliconProfile(
        chip_family=chip_family,
        chip_variant=chip_variant,
        cpu_cores=hardware_info.cpu_cores,
        performance_cores=hardware_info.cpu_cores // 2,
        efficiency_cores=hardware_info.cpu_cores // 2,
        gpu_cores=hardware_info.gpu_cores or 0,
        neural_engine_cores=hardware_info.neural_engine_cores or 0,
        memory_gb=int(hardware_info.memory_gb),
        memory_bandwidth_gbps=100,  # Default estimate
        supports_ray_tracing=chip_family >= "M3"  # M3 and newer support ray tracing
    )

def detect_hardware() -> AppleSiliconProfile:
    """
    Detect the current hardware and return the matching profile.
    
    Returns:
        AppleSiliconProfile for the current hardware
    """
    # Use the existing hardware detection from model_selector
    hardware_info = get_hardware_info()
    
    # Map to an AppleSiliconProfile
    return map_hardware_info_to_profile(hardware_info)

def get_optimal_configuration(profile: AppleSiliconProfile, 
                              task_type: str,
                              prioritize_speed: bool = False,
                              prioritize_quality: bool = False) -> Dict:
    """
    Get the optimal configuration for a given hardware profile and task.
    
    Args:
        profile: Hardware profile
        task_type: Type of task (inference, fine-tuning, etc.)
        prioritize_speed: Whether to prioritize speed over quality
        prioritize_quality: Whether to prioritize quality over speed
        
    Returns:
        Dictionary with optimal configuration parameters
    """
    # Implementation to determine optimal configuration
    pass