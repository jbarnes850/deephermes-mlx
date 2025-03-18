"""
Main script for running DeepHermes-3-Mistral-24B MLX inference.
"""
import argparse
from typing import Dict, Any, Optional

from model import load_model
from inference import run_inference


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with DeepHermes-3-Mistral-24B MLX model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16",
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
        default="You are DeepHermes, a helpful AI assistant.",
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
        help="Enable reasoning mode (adds 'Reasoning:' to system prompt)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run inference."""
    args = parse_args()
    
    # Configure tokenizer
    tokenizer_config: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code
    }
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        model_path=args.model,
        trust_remote_code=args.trust_remote_code,
        tokenizer_config=tokenizer_config
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
        system_prompt += " Please think through this step by step and show your reasoning."
    
    # Run inference
    print("\n--- Model Output ---\n")
    run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
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
