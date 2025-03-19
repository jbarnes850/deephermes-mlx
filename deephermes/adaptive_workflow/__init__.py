"""
Adaptive ML Workflow for DeepHermes MLX.

This module provides a framework for intelligently configuring the ML pipeline
based on hardware capabilities, creating a seamless "it just works" experience.
"""

from .hardware_profiles import AppleSiliconProfile, detect_hardware
from .config_manager import AdaptiveWorkflowConfig
from .cli import main as cli_main

__all__ = ["AppleSiliconProfile", "detect_hardware", "AdaptiveWorkflowConfig", "cli_main"]
