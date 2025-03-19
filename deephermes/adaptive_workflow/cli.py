"""
Command-line interface for the Adaptive ML Workflow.

This module provides a unified CLI interface for configuring and executing
the adaptive workflow based on hardware capabilities.
"""

import argparse
import sys
import json
import os
from typing import Dict, Any, Optional
from .config_manager import AdaptiveWorkflowConfig
from .hardware_profiles import detect_hardware, AppleSiliconProfile
from ..model_selector.hardware_detection import get_hardware_info

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DeepHermes Adaptive ML Workflow")
    
    parser.add_argument(
        "--workflow",
        choices=["general", "content_creation", "coding", "research"],
        default="general",
        help="Workflow type to configure"
    )
    
    parser.add_argument(
        "--prioritize",
        choices=["speed", "quality", "balanced"],
        default="balanced",
        help="Performance priority"
    )
    
    parser.add_argument(
        "--max-memory",
        type=float,
        default=80.0,
        help="Maximum percentage of memory to use (0-100)"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to file"
    )
    
    parser.add_argument(
        "--load-config",
        type=str,
        help="Load configuration from file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration in JSON format"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show hardware and performance dashboard"
    )
    
    return parser.parse_args()

def print_hardware_info(hardware_profile: AppleSiliconProfile, json_output: bool = False) -> None:
    """
    Print hardware information in a user-friendly format.
    
    Args:
        hardware_profile: Hardware profile to print
        json_output: Whether to output in JSON format
    """
    if json_output:
        # For JSON output, we'll include this in the main output
        return
    
    print("\n===== Hardware Information =====")
    print(f"Chip: {hardware_profile.chip_family} {hardware_profile.chip_variant}")
    print(f"CPU: {hardware_profile.cpu_cores} cores ({hardware_profile.performance_cores} performance, {hardware_profile.efficiency_cores} efficiency)")
    print(f"GPU: {hardware_profile.gpu_cores} cores")
    print(f"Neural Engine: {hardware_profile.neural_engine_cores} cores")
    print(f"Memory: {hardware_profile.memory_gb} GB")
    print(f"Memory Bandwidth: {hardware_profile.memory_bandwidth_gbps} GB/s")
    print(f"Ray Tracing: {'Supported' if hardware_profile.supports_ray_tracing else 'Not supported'}")
    print(f"Compute Power Score: {hardware_profile.total_compute_power:.1f}")
    print()

def print_model_config(model_config: Dict[str, Any], json_output: bool = False) -> None:
    """
    Print model configuration in a user-friendly format.
    
    Args:
        model_config: Model configuration to print
        json_output: Whether to output in JSON format
    """
    if json_output:
        # For JSON output, we'll include this in the main output
        return
    
    print("===== Model Configuration =====")
    print(f"Model: {model_config['model_id']}")
    print(f"Quantization: {model_config['quantization'] or 'None (full precision)'}")
    print(f"Lazy Loading: {'Enabled' if model_config['lazy_load'] else 'Disabled'}")
    print(f"Max Tokens: {model_config['max_tokens']}")
    print(f"Memory Required: {model_config['memory_required_gb']:.1f} GB")
    print(f"Recommendation Confidence: {model_config.get('recommendation_confidence', 0.8):.2f}")
    print(f"Reason: {model_config.get('recommendation_reason', 'No reason provided')}")
    print()

def print_workflow_config(config: AdaptiveWorkflowConfig, json_output: bool = False) -> None:
    """
    Print the full workflow configuration.
    
    Args:
        config: Workflow configuration to print
        json_output: Whether to output in JSON format
    """
    if json_output:
        # Output as JSON
        print(json.dumps(config.get_full_config(), indent=2))
        return
    
    # Print hardware and model info
    print_hardware_info(config.hardware_profile)
    print_model_config(config.model_config)
    
    # Print workflow info
    print("===== Workflow Configuration =====")
    print(f"Workflow Type: {config.workflow_type}")
    print(f"Performance Target: {'Speed' if config.prioritize_speed else 'Quality' if config.prioritize_quality else 'Balanced'}")
    print(f"Max Memory Usage: {config.max_memory_usage_pct:.1f}%")
    print()
    
    # Print fine-tuning config
    print("===== Fine-tuning Configuration =====")
    for key, value in config.fine_tuning_config.items():
        print(f"{key}: {value}")
    print()
    
    # Print serving config
    print("===== Serving Configuration =====")
    for key, value in config.serving_config.items():
        print(f"{key}: {value}")
    print()
    
    # Print integration config
    print("===== Integration Configuration =====")
    for key, value in config.integration_config.items():
        print(f"{key}: {value}")
    print()

def main() -> None:
    """Main entry point for the adaptive workflow CLI."""
    args = parse_args()
    
    if args.dashboard:
        # Show dashboard
        try:
            from .dashboard import show_dashboard
            show_dashboard()
        except ImportError:
            print("Dashboard functionality not available. Please install the required dependencies.")
            print("pip install matplotlib pandas")
        return
    
    # Create or load configuration
    if args.load_config:
        try:
            config = AdaptiveWorkflowConfig.load_config(args.load_config)
            if not args.json:
                print(f"Configuration loaded from {args.load_config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    else:
        # Detect hardware
        try:
            hardware_profile = detect_hardware()
            
            # Create configuration
            config = AdaptiveWorkflowConfig(
                hardware_profile=hardware_profile,
                workflow_type=args.workflow,
                prioritize_speed=args.prioritize == "speed",
                prioritize_quality=args.prioritize == "quality",
                max_memory_usage_pct=args.max_memory,
            )
        except Exception as e:
            print(f"Error creating configuration: {e}")
            sys.exit(1)
    
    # Save configuration if requested
    if args.save_config:
        try:
            config.save_config(args.save_config)
            if not args.json:
                print(f"Configuration saved to {args.save_config}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
            sys.exit(1)
    
    # Print configuration
    print_workflow_config(config, args.json)
    
if __name__ == "__main__":
    main()