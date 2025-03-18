"""
Command-line interface for DeepHermes MLX.
"""
import argparse
from typing import Dict, Any, Optional, List

from deephermes.core.model import load_model
from deephermes.core.inference import run_inference
from deephermes.core.utils import get_reasoning_prompt, get_default_system_prompt


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with DeepHermes MLX models")
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16",
        help="Model path on Hugging Face Hub"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to use for generation"
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
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
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
    parser.add_argument(
        "--auto-config",
        action="store_true",
        help="Automatically configure model based on hardware capabilities"
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
    
    return parser.parse_args()


def main() -> None:
    """Main function to run inference."""
    args = parse_args()
    
    # Check if auto-config is enabled
    if args.auto_config:
        try:
            from deephermes.model_selector.integration import get_optimal_configuration
            config = get_optimal_configuration()
            
            # Apply configuration if available
            if config:
                if not args.model and 'model' in config:
                    args.model = config['model']
                if not args.quantize and 'quantize' in config:
                    args.quantize = config['quantize']
                if not args.lazy_load and config.get('lazy_load', False):
                    args.lazy_load = True
                
                print(f"Using auto-configured settings: {config}")
        except ImportError:
            print("Model selector not available, using default configuration")
    
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
    
    # Get user prompt if not provided
    prompt = args.prompt
    if not prompt:
        print("\nEnter your prompt (press Ctrl+D or type 'exit' to quit):")
        user_input = []
        try:
            while True:
                line = input()
                if line.strip().lower() == "exit":
                    break
                user_input.append(line)
        except EOFError:
            pass
        prompt = "\n".join(user_input)
    
    # Modify system prompt for reasoning if enabled
    system_prompt = args.system_prompt
    if args.reasoning:
        reasoning_prompt = get_reasoning_prompt(args.reasoning_depth)
        system_prompt += reasoning_prompt
    
    # Format messages in chat format
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})
    
    # Run inference
    print("\n--- Model Output ---\n")
    run_inference(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
        verbose=True,
        max_kv_size=args.max_kv_size
    )
    print("\n--- End of Output ---\n")


if __name__ == "__main__":
    main()
