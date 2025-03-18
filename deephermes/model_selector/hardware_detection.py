"""
Hardware detection utilities for the DeepHermes Model Selector.

This module provides functions for detecting hardware capabilities
and available memory to inform model selection decisions.
"""

from typing import Dict, Any, Tuple, Optional
import platform
import subprocess
import re
import os
import psutil
import json
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Information about the system hardware."""
    device_name: str
    chip_type: str
    memory_gb: float
    cpu_cores: int
    gpu_cores: Optional[int] = None
    neural_engine_cores: Optional[int] = None
    is_apple_silicon: bool = False


def get_mac_chip_info() -> Dict[str, Any]:
    """
    Get detailed information about Apple Silicon chips.
    
    Returns:
        Dictionary with chip information
    """
    chip_info = {
        "chip_type": "Unknown",
        "gpu_cores": None,
        "neural_engine_cores": None,
        "is_apple_silicon": False
    }
    
    try:
        # Get chip model
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        chip_model = result.stdout.strip()
        
        # Check if it's Apple Silicon
        if "Apple" in chip_model:
            chip_info["is_apple_silicon"] = True
            
            # Extract chip type (M1, M2, M3, etc.)
            match = re.search(r'Apple (M\d+)(?:\s+(\w+))?', chip_model)
            if match:
                chip_type = match.group(1)
                variant = match.group(2) if match.group(2) else ""
                chip_info["chip_type"] = f"{chip_type} {variant}".strip()
                
                # Estimate GPU and Neural Engine cores based on chip type
                if "M1" in chip_type:
                    if "Max" in variant:
                        chip_info["gpu_cores"] = 32
                        chip_info["neural_engine_cores"] = 16
                    elif "Pro" in variant:
                        chip_info["gpu_cores"] = 16
                        chip_info["neural_engine_cores"] = 16
                    elif "Ultra" in variant:
                        chip_info["gpu_cores"] = 64
                        chip_info["neural_engine_cores"] = 32
                    else:
                        chip_info["gpu_cores"] = 8
                        chip_info["neural_engine_cores"] = 16
                elif "M2" in chip_type:
                    if "Max" in variant:
                        chip_info["gpu_cores"] = 38
                        chip_info["neural_engine_cores"] = 16
                    elif "Pro" in variant:
                        chip_info["gpu_cores"] = 19
                        chip_info["neural_engine_cores"] = 16
                    elif "Ultra" in variant:
                        chip_info["gpu_cores"] = 76
                        chip_info["neural_engine_cores"] = 32
                    else:
                        chip_info["gpu_cores"] = 10
                        chip_info["neural_engine_cores"] = 16
                elif "M3" in chip_type:
                    if "Max" in variant:
                        chip_info["gpu_cores"] = 40
                        chip_info["neural_engine_cores"] = 16
                    elif "Pro" in variant:
                        chip_info["gpu_cores"] = 19
                        chip_info["neural_engine_cores"] = 16
                    elif "Ultra" in variant:
                        chip_info["gpu_cores"] = 80
                        chip_info["neural_engine_cores"] = 32
                    else:
                        chip_info["gpu_cores"] = 10
                        chip_info["neural_engine_cores"] = 16
                elif "M4" in chip_type:
                    if "Max" in variant:
                        chip_info["gpu_cores"] = 40
                        chip_info["neural_engine_cores"] = 32
                    elif "Pro" in variant:
                        chip_info["gpu_cores"] = 20
                        chip_info["neural_engine_cores"] = 32
                    elif "Ultra" in variant:
                        chip_info["gpu_cores"] = 80
                        chip_info["neural_engine_cores"] = 64
                    else:
                        chip_info["gpu_cores"] = 10
                        chip_info["neural_engine_cores"] = 32
    except Exception as e:
        print(f"Error getting chip info: {e}")
    
    return chip_info


def get_available_memory() -> float:
    """
    Get the amount of available memory in GB.
    
    Returns:
        Available memory in GB
    """
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)  # Convert to GB
    except Exception as e:
        print(f"Error getting available memory: {e}")
        return 0.0


def get_total_memory() -> float:
    """
    Get the total amount of system memory in GB.
    
    Returns:
        Total memory in GB
    """
    try:
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)  # Convert to GB
    except Exception as e:
        print(f"Error getting total memory: {e}")
        return 0.0


def get_cpu_cores() -> int:
    """
    Get the number of CPU cores.
    
    Returns:
        Number of CPU cores
    """
    try:
        return os.cpu_count() or 0
    except Exception as e:
        print(f"Error getting CPU cores: {e}")
        return 0


def get_hardware_info() -> HardwareInfo:
    """
    Get comprehensive hardware information.
    
    Returns:
        HardwareInfo object with system details
    """
    system = platform.system()
    device_name = platform.node()
    
    if system == "Darwin":
        # macOS-specific detection
        chip_info = get_mac_chip_info()
        
        return HardwareInfo(
            device_name=device_name,
            chip_type=chip_info["chip_type"],
            memory_gb=get_total_memory(),
            cpu_cores=get_cpu_cores(),
            gpu_cores=chip_info["gpu_cores"],
            neural_engine_cores=chip_info["neural_engine_cores"],
            is_apple_silicon=chip_info["is_apple_silicon"]
        )
    else:
        # Generic detection for other platforms
        return HardwareInfo(
            device_name=device_name,
            chip_type=platform.processor() or "Unknown",
            memory_gb=get_total_memory(),
            cpu_cores=get_cpu_cores(),
            is_apple_silicon=False
        )


def save_hardware_info(file_path: str = "hardware_info.json") -> None:
    """
    Save hardware information to a JSON file.
    
    Args:
        file_path: Path to save the hardware info
    """
    hardware_info = get_hardware_info()
    
    # Convert to dictionary
    info_dict = {
        "device_name": hardware_info.device_name,
        "chip_type": hardware_info.chip_type,
        "memory_gb": hardware_info.memory_gb,
        "cpu_cores": hardware_info.cpu_cores,
        "gpu_cores": hardware_info.gpu_cores,
        "neural_engine_cores": hardware_info.neural_engine_cores,
        "is_apple_silicon": hardware_info.is_apple_silicon
    }
    
    with open(file_path, "w") as f:
        json.dump(info_dict, f, indent=2)
    
    print(f"Hardware information saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    info = get_hardware_info()
    print(f"Device: {info.device_name}")
    print(f"Chip: {info.chip_type}")
    print(f"Memory: {info.memory_gb:.1f} GB")
    print(f"CPU Cores: {info.cpu_cores}")
    
    if info.is_apple_silicon:
        print(f"GPU Cores: {info.gpu_cores}")
        print(f"Neural Engine Cores: {info.neural_engine_cores}")
    
    print(f"Available Memory: {get_available_memory():.1f} GB")
