"""
Interactive chat script for DeepHermes MLX inference.
"""
from typing import List, Dict, Any, Optional
import argparse
import os
import sys

from deephermes.core.model import load_model
from deephermes.core.inference import run_inference
from deephermes.core.utils import get_reasoning_prompt, get_default_system_prompt

# Add model selector integration
try:
    from deephermes.model_selector.integration import get_model_args
    MODEL_SELECTOR_AVAILABLE = True
except ImportError:
    MODEL_SELECTOR_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive chat with DeepHermes MLX models")
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
        help="Model path on Hugging Face Hub"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=get_default_system_prompt(),
        help="System prompt to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in tokenizer"
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size (for long context)"
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable DeepHermes reasoning mode (adds reasoning instruction to system prompt)"
    )
    parser.add_argument(
        "--reasoning-depth",
        type=str,
        choices=["basic", "deep", "expert"],
        default="deep",
        help="Depth of reasoning when reasoning mode is enabled"
    )
    # Add memory-efficient options
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Quantize model to reduce memory usage (4bit or 8bit)"
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Load model weights lazily to reduce memory usage"
    )
    parser.add_argument(
        "--auto-config",
        action="store_true",
        help="Automatically configure model based on hardware capabilities"
    )
    
    return parser.parse_args()


def print_help() -> None:
    """Print help information for chat commands."""
    print("\nAvailable commands:")
    print("  exit                  - Exit the chat")
    print("  clear                 - Clear chat history")
    print("  system <prompt>       - Change the system prompt")
    print("  reasoning <on|off>    - Toggle reasoning mode")
    print("  help                  - Show this help message")
    
    # Add benchmark and model selector commands if available
    benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
    if os.path.exists(benchmark_path):
        print("\nBenchmark commands:")
        print("  benchmark            - Run benchmark with default settings")
        print("  benchmark <options>  - Run benchmark with custom options")
        print("                          (e.g., 'benchmark --datasets mmlu --num-samples 10')")
    
    if MODEL_SELECTOR_AVAILABLE:
        print("\nModel selector commands:")
        print("  recommend            - Get model recommendations based on hardware")
        print("  switch <model>       - Switch to a different model")
        print("                          (e.g., 'switch 3B', 'switch 8B', 'switch 24B')")


def main() -> None:
    """Main function to run interactive chat."""
    args = parse_args()
    
    # Apply automatic configuration if requested
    if args.auto_config and MODEL_SELECTOR_AVAILABLE:
        args = get_model_args(args)
        print(f"Auto-configured model: {args.model}")
        print(f"Quantization: {args.quantize if args.quantize else 'None'}")
        print(f"Lazy loading: {'Enabled' if args.lazy_load else 'Disabled'}")
    
    # Configure tokenizer
    tokenizer_config: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code
    }
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        model_path=args.model,
        quantize=args.quantize,
        lazy_load=args.lazy_load,
        trust_remote_code=args.trust_remote_code,
        tokenizer_config=tokenizer_config,
        verbose=True
    )
    
    # Modify system prompt for reasoning if enabled
    system_prompt = args.system_prompt
    if args.reasoning:
        reasoning_prompt = get_reasoning_prompt(args.reasoning_depth)
        system_prompt += reasoning_prompt
    
    # Initialize chat history
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    print("\n=== DeepHermes Chat ===")
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
            print_help()
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
            if toggle == "on":
                # Update system prompt with reasoning
                base_system_prompt = system_prompt
                # Remove any existing reasoning prompt
                reasoning_text = get_reasoning_prompt()
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
                reasoning_text = get_reasoning_prompt()
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
        # Handle benchmark command
        elif user_input.lower().startswith("benchmark"):
            benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "run_benchmark.py")
            if os.path.exists(benchmark_path):
                # Extract benchmark options
                if user_input.lower() == "benchmark":
                    benchmark_cmd = f"python {benchmark_path}"
                else:
                    options = user_input[len("benchmark"):].strip()
                    benchmark_cmd = f"python {benchmark_path} {options}"
                
                print(f"Running benchmark: {benchmark_cmd}")
                print("This may take some time. Please wait...")
                
                # Run benchmark in a subprocess
                import subprocess
                try:
                    subprocess.run(benchmark_cmd, shell=True, check=True)
                    print("Benchmark completed.")
                except subprocess.CalledProcessError as e:
                    print(f"Benchmark failed with error: {e}")
            else:
                print("Benchmark module not found. Please make sure the benchmarks directory exists.")
            continue
        # Handle model recommendation command
        elif user_input.lower() == "recommend" and MODEL_SELECTOR_AVAILABLE:
            model_selector_path = os.path.join(os.path.dirname(__file__), "model_selector", "cli.py")
            if os.path.exists(model_selector_path):
                print("Getting model recommendations...")
                
                # Run model selector in a subprocess
                import subprocess
                try:
                    subprocess.run(f"python {model_selector_path}", shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Model selector failed with error: {e}")
            else:
                print("Model selector module not found.")
            continue
        # Handle model switching command
        elif user_input.lower().startswith("switch ") and MODEL_SELECTOR_AVAILABLE:
            model_size = user_input[7:].strip().lower()
            
            # Map model size to model ID
            model_map = {
                "3b": "mlx-community/DeepHermes-3-Llama-3-3B-Preview-bf16",
                "8b": "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
                "24b": "mlx-community/DeepHermes-3-Mistral-24B-Preview-bf16"
            }
            
            if model_size in model_map:
                new_model_id = model_map[model_size]
                print(f"Switching to {new_model_id}...")
                
                # Reload model
                try:
                    model, tokenizer = load_model(
                        model_path=new_model_id,
                        quantize=args.quantize,
                        lazy_load=args.lazy_load,
                        trust_remote_code=args.trust_remote_code,
                        tokenizer_config=tokenizer_config,
                        verbose=True
                    )
                    
                    # Update args
                    args.model = new_model_id
                    
                    print(f"Successfully switched to {new_model_id}")
                except Exception as e:
                    print(f"Failed to switch model: {e}")
            else:
                print(f"Unknown model size: {model_size}")
                print("Available options: 3b, 8b, 24b")
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("\nDeepHermes: ", end="", flush=True)
        response = run_inference(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Add assistant message to history
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
