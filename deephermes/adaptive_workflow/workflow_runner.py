"""
Workflow runner for the Adaptive ML Workflow.

This module provides functionality to execute different workflow templates
based on the user's requirements and hardware capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import json
import logging
from pathlib import Path

from .config_manager import AdaptiveWorkflowConfig
from .workflow_templates import get_workflow_template, WorkflowTemplate
from ..model_selector.hardware_detection import get_hardware_info
from ..core.model import load_model
from ..core.inference import run_inference


class WorkflowRunner:
    """Runner for executing adaptive ML workflows."""
    
    def __init__(self, 
                 workflow_type: str = "general",
                 config_path: Optional[str] = None,
                 prioritize_speed: bool = False,
                 prioritize_quality: bool = False,
                 max_memory_usage_pct: float = 80.0,
                 verbose: bool = True):
        """
        Initialize the workflow runner.
        
        Args:
            workflow_type: Type of workflow to run
            config_path: Path to a saved configuration file, or None to generate a new one
            prioritize_speed: Whether to prioritize speed over quality
            prioritize_quality: Whether to prioritize quality over speed
            max_memory_usage_pct: Maximum percentage of memory to use
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = AdaptiveWorkflowConfig.load_config(config_path)
            if workflow_type != self.config.workflow_type:
                self.logger.info(f"Switching workflow type from {self.config.workflow_type} to {workflow_type}")
                self.config = AdaptiveWorkflowConfig(
                    hardware_profile=self.config.hardware_profile,
                    workflow_type=workflow_type,
                    prioritize_speed=prioritize_speed,
                    prioritize_quality=prioritize_quality,
                    max_memory_usage_pct=max_memory_usage_pct
                )
        else:
            self.config = AdaptiveWorkflowConfig(
                workflow_type=workflow_type,
                prioritize_speed=prioritize_speed,
                prioritize_quality=prioritize_quality,
                max_memory_usage_pct=max_memory_usage_pct
            )
        
        # Get the workflow template
        self.workflow_template = get_workflow_template(workflow_type)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger."""
        logger = logging.getLogger("workflow_runner")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def initialize(self) -> None:
        """Initialize the workflow by loading the model and tokenizer."""
        if self.verbose:
            self.logger.info(f"Initializing {self.workflow_template.name} workflow")
            self.logger.info(f"Model: {self.config.model_config['model_id']}")
            self.logger.info(f"Quantization: {self.config.model_config['quantization']}")
            self.logger.info(f"Lazy loading: {self.config.model_config['lazy_load']}")
        
        # Configure tokenizer
        tokenizer_config = {
            "trust_remote_code": True  # Adjust as needed
        }
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model(
            model_path=self.config.model_config["model_id"],
            quantize=self.config.model_config["quantization"],
            lazy_load=self.config.model_config["lazy_load"],
            trust_remote_code=True,
            tokenizer_config=tokenizer_config,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info("Model and tokenizer loaded successfully")
    
    def run_chat(self) -> None:
        """Run the chat interface for the workflow."""
        if not self.model or not self.tokenizer:
            self.initialize()
        
        # Get system prompt from workflow template
        system_prompt = self.workflow_template.get_full_system_prompt()
        
        # Initialize chat history
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        print(f"\n=== DeepHermes {self.workflow_template.name.capitalize()} Chat ===")
        print(f"Workflow: {self.workflow_template.description}")
        print("Type 'exit' to quit, 'clear' to clear history, 'system <prompt>' to change system prompt,")
        print("or 'reasoning <on|off>' to toggle reasoning mode. Type 'help' for more commands.")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for special commands
            if user_input.lower() == "exit":
                print("Exiting chat...")
                break
            elif user_input.lower() == "clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print("Chat history cleared.")
                continue
            elif user_input.lower() == "help":
                self._print_help()
                continue
            elif user_input.lower().startswith("system "):
                new_system_prompt = user_input[7:].strip()
                # Replace system message or add if not present
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = new_system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": new_system_prompt})
                system_prompt = new_system_prompt
                print(f"System prompt updated: {new_system_prompt}")
                continue
            elif user_input.lower().startswith("reasoning "):
                toggle = user_input[10:].strip().lower()
                from ..core.utils import get_reasoning_prompt
                
                if toggle == "on":
                    # Update system prompt with reasoning
                    base_system_prompt = system_prompt
                    # Remove any existing reasoning prompt
                    reasoning_text = get_reasoning_prompt(self.workflow_template.reasoning_depth)
                    if base_system_prompt.endswith(reasoning_text):
                        print("Reasoning is already enabled.")
                        continue
                    
                    # Add reasoning prompt
                    system_prompt = base_system_prompt + reasoning_text
                    
                    # Update in messages
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    
                    print("Reasoning mode enabled.")
                    continue
                elif toggle == "off":
                    # Remove reasoning prompt from system prompt
                    reasoning_text = get_reasoning_prompt(self.workflow_template.reasoning_depth)
                    if system_prompt.endswith(reasoning_text):
                        system_prompt = system_prompt[:-len(reasoning_text)]
                        
                        # Update in messages
                        if messages and messages[0]["role"] == "system":
                            messages[0]["content"] = system_prompt
                        else:
                            messages.insert(0, {"role": "system", "content": system_prompt})
                        
                        print("Reasoning mode disabled.")
                    else:
                        print("Reasoning is already disabled.")
                    continue
                else:
                    print("Invalid reasoning option. Use 'on' or 'off'.")
                    continue
            elif user_input.lower().startswith("workflow "):
                new_workflow = user_input[9:].strip().lower()
                try:
                    # Change workflow template
                    self.workflow_template = get_workflow_template(new_workflow)
                    self.config = AdaptiveWorkflowConfig(
                        hardware_profile=self.config.hardware_profile,
                        workflow_type=new_workflow,
                        prioritize_speed=self.config.prioritize_speed,
                        prioritize_quality=self.config.prioritize_quality,
                        max_memory_usage_pct=self.config.max_memory_usage_pct
                    )
                    
                    # Update system prompt
                    system_prompt = self.workflow_template.get_full_system_prompt()
                    
                    # Update in messages
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        messages = [{"role": "system", "content": system_prompt}]
                    
                    print(f"Switched to {new_workflow} workflow: {self.workflow_template.description}")
                except ValueError as e:
                    print(f"Error: {e}")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nDeepHermes: ", end="", flush=True)
            response = run_inference(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=messages,
                max_tokens=self.config.serving_config.get("max_tokens_per_request", 4096),
                temperature=self.config.serving_config.get("temperature", 0.7),
                top_p=self.config.serving_config.get("top_p", 0.9)
            )
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": response})
    
    def _print_help(self) -> None:
        """Print help information for chat commands."""
        print("\nAvailable commands:")
        print("  exit                  - Exit the chat")
        print("  clear                 - Clear chat history")
        print("  system <prompt>       - Change the system prompt")
        print("  reasoning <on|off>    - Toggle reasoning mode")
        print("  workflow <type>       - Switch to a different workflow type")
        print("                          Available types: general, content_creation, coding, research")
        print("  help                  - Show this help message")
    
    def run_langchain_integration(self) -> None:
        """Run the LangChain integration for the workflow."""
        if not self.model or not self.tokenizer:
            self.initialize()
        
        try:
            from ..integrate.langchain import setup_langchain_agent
            
            print(f"\n=== DeepHermes {self.workflow_template.name.capitalize()} with LangChain ===")
            print(f"Workflow: {self.workflow_template.description}")
            print("Setting up LangChain integration...")
            
            # Get system prompt from workflow template
            system_prompt = self.workflow_template.get_full_system_prompt()
            
            # Setup LangChain agent
            agent = setup_langchain_agent(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=system_prompt,
                temperature=self.config.serving_config.get("temperature", 0.7),
                max_tokens=self.config.serving_config.get("max_tokens_per_request", 4096)
            )
            
            print("LangChain integration ready. Type 'exit' to quit.")
            
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == "exit":
                    print("Exiting LangChain integration...")
                    break
                
                # Run agent
                print("\nDeepHermes: ", end="", flush=True)
                response = agent.run(user_input)
                print(response)
                
        except ImportError:
            print("LangChain integration is not available. Please install the required dependencies.")
            print("You can install them with: pip install langchain")
    
    def save_config(self, file_path: str) -> None:
        """Save the configuration to a file."""
        self.config.save_config(file_path)
        if self.verbose:
            self.logger.info(f"Configuration saved to {file_path}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response to a prompt without interactive chat.
        
        Args:
            prompt: The prompt to generate a response for
            
        Returns:
            The generated response
        """
        if not self.model or not self.tokenizer:
            self.initialize()
        
        # Get system prompt from workflow template
        system_prompt = self.workflow_template.get_full_system_prompt()
        
        # Initialize messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = run_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=messages,
            max_tokens=self.config.serving_config.get("max_tokens_per_request", 4096),
            temperature=self.config.serving_config.get("temperature", 0.7),
            top_p=self.config.serving_config.get("top_p", 0.9)
        )
        
        return response
    
    @classmethod
    def from_config_file(cls, config_path: str, verbose: bool = True) -> 'WorkflowRunner':
        """Create a workflow runner from a configuration file."""
        return cls(config_path=config_path, verbose=verbose)


def main() -> None:
    """Main function to run the workflow runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run adaptive ML workflows")
    parser.add_argument(
        "--workflow",
        type=str,
        default="general",
        choices=["general", "content_creation", "coding", "research"],
        help="Type of workflow to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a saved configuration file"
    )
    parser.add_argument(
        "--prioritize-speed",
        action="store_true",
        help="Prioritize speed over quality"
    )
    parser.add_argument(
        "--prioritize-quality",
        action="store_true",
        help="Prioritize quality over speed"
    )
    parser.add_argument(
        "--max-memory-usage",
        type=float,
        default=80.0,
        help="Maximum percentage of memory to use"
    )
    parser.add_argument(
        "--langchain",
        action="store_true",
        help="Use LangChain integration"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save the configuration to a file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Create workflow runner
    runner = WorkflowRunner(
        workflow_type=args.workflow,
        config_path=args.config,
        prioritize_speed=args.prioritize_speed,
        prioritize_quality=args.prioritize_quality,
        max_memory_usage_pct=args.max_memory_usage,
        verbose=args.verbose
    )
    
    # Save configuration if requested
    if args.save_config:
        runner.save_config(args.save_config)
    
    # Run workflow
    if args.langchain:
        runner.run_langchain_integration()
    else:
        runner.run_chat()


if __name__ == "__main__":
    main()
