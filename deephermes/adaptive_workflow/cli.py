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
from .workflow_templates import get_workflow_template, WORKFLOW_TEMPLATES
from .workflow_runner import WorkflowRunner
from ..model_selector.hardware_detection import get_hardware_info

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DeepHermes Adaptive ML Workflow")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Configure command
    configure_parser = subparsers.add_parser("configure", help="Configure the workflow")
    configure_parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_TEMPLATES.keys()),
        default="general",
        help="Workflow type to configure"
    )
    configure_parser.add_argument(
        "--prioritize",
        choices=["speed", "quality", "balanced"],
        default="balanced",
        help="Performance priority"
    )
    configure_parser.add_argument(
        "--max-memory",
        type=float,
        default=80.0,
        help="Maximum percentage of memory to use (0-100)"
    )
    configure_parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to file"
    )
    configure_parser.add_argument(
        "--load-config",
        type=str,
        help="Load configuration from file"
    )
    configure_parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration in JSON format"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_TEMPLATES.keys()),
        default="general",
        help="Workflow type to run"
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from file"
    )
    run_parser.add_argument(
        "--prioritize",
        choices=["speed", "quality", "balanced"],
        default="balanced",
        help="Performance priority"
    )
    run_parser.add_argument(
        "--max-memory",
        type=float,
        default=80.0,
        help="Maximum percentage of memory to use (0-100)"
    )
    run_parser.add_argument(
        "--langchain",
        action="store_true",
        help="Use LangChain integration for enhanced reasoning capabilities"
    )
    run_parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to file before running"
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show hardware and performance dashboard")
    
    # List workflows command
    list_parser = subparsers.add_parser("list", help="List available workflows")
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # For backwards compatibility, add top-level arguments
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_TEMPLATES.keys()),
        help="Workflow type to configure (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--prioritize",
        choices=["speed", "quality", "balanced"],
        help="Performance priority (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        help="Maximum percentage of memory to use (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to file (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--load-config",
        type=str,
        help="Load configuration from file (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration in JSON format (deprecated, use configure or run subcommands)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show hardware and performance dashboard (deprecated, use dashboard subcommand)"
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

def print_workflow_template(template_name: str, json_output: bool = False) -> None:
    """
    Print information about a workflow template.
    
    Args:
        template_name: Name of the workflow template
        json_output: Whether to output in JSON format
    """
    template = get_workflow_template(template_name)
    
    if json_output:
        template_data = {
            "name": template.name,
            "description": template.description,
            "system_prompt": template.system_prompt,
            "reasoning_enabled": template.reasoning_enabled,
            "reasoning_depth": template.reasoning_depth,
        }
        print(json.dumps(template_data, indent=2))
        return
    
    print(f"\n===== {template.name.capitalize()} Workflow =====")
    print(f"Description: {template.description}")
    print(f"Reasoning: {'Enabled (' + template.reasoning_depth + ')' if template.reasoning_enabled else 'Disabled'}")
    print("\nSystem Prompt:")
    print(f"{template.system_prompt}")
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
    print(f"Description: {config.workflow_template.description}")
    print(f"Performance Target: {'Speed' if config.prioritize_speed else 'Quality' if config.prioritize_quality else 'Balanced'}")
    print(f"Max Memory Usage: {config.max_memory_usage_pct:.1f}%")
    print()
    
    # Print system prompt
    print("===== System Prompt =====")
    print(config.workflow_template.system_prompt)
    print()
    
    # Print reasoning settings
    print("===== Reasoning Settings =====")
    print(f"Enabled: {config.workflow_template.reasoning_enabled}")
    print(f"Depth: {config.workflow_template.reasoning_depth}")
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

def list_workflows(json_output: bool = False) -> None:
    """
    List all available workflow templates.
    
    Args:
        json_output: Whether to output in JSON format
    """
    if json_output:
        workflows = {}
        for name, template in WORKFLOW_TEMPLATES.items():
            workflows[name] = {
                "name": template.name,
                "description": template.description,
                "reasoning_enabled": template.reasoning_enabled,
                "reasoning_depth": template.reasoning_depth,
            }
        print(json.dumps(workflows, indent=2))
        return
    
    print("\n===== Available Workflows =====")
    for name, template in WORKFLOW_TEMPLATES.items():
        print(f"{name}: {template.description}")
        print(f"  Reasoning: {'Enabled (' + template.reasoning_depth + ')' if template.reasoning_enabled else 'Disabled'}")
        print()

def handle_configure_command(args: argparse.Namespace) -> None:
    """Handle the configure command."""
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

def handle_run_command(args: argparse.Namespace) -> None:
    """Handle the run command."""
    try:
        # Create workflow runner
        runner = WorkflowRunner(
            workflow_type=args.workflow,
            config_path=args.config,
            prioritize_speed=args.prioritize == "speed",
            prioritize_quality=args.prioritize == "quality",
            max_memory_usage_pct=args.max_memory,
            verbose=args.verbose
        )
        
        # Save configuration if requested
        if args.save_config:
            runner.save_config(args.save_config)
            if args.verbose:
                print(f"Configuration saved to {args.save_config}")
        
        # Run workflow
        if args.langchain:
            runner.run_langchain_integration()
        else:
            runner.run_chat()
    except Exception as e:
        print(f"Error running workflow: {e}")
        sys.exit(1)

def handle_dashboard_command() -> None:
    """Handle the dashboard command."""
    try:
        from .dashboard import show_dashboard
        show_dashboard()
    except ImportError:
        print("Dashboard functionality not available. Please install the required dependencies.")
        print("pip install matplotlib pandas")

def handle_list_command(args: argparse.Namespace) -> None:
    """Handle the list command."""
    list_workflows(args.json)

def main() -> None:
    """Main entry point for the adaptive workflow CLI."""
    args = parse_args()
    
    # Handle subcommands
    if args.command == "configure":
        handle_configure_command(args)
    elif args.command == "run":
        handle_run_command(args)
    elif args.command == "dashboard":
        handle_dashboard_command()
    elif args.command == "list":
        handle_list_command(args)
    else:
        # Backward compatibility with old CLI interface
        if args.dashboard:
            handle_dashboard_command()
            return
        
        # If no command specified but workflow is, assume configure
        if any([args.workflow, args.prioritize, args.max_memory, args.save_config, args.load_config]):
            handle_configure_command(args)
        else:
            # Print help if no command or arguments
            print("Please specify a command: configure, run, dashboard, or list")
            print("Run with --help for more information")
            sys.exit(1)

if __name__ == "__main__":
    main()