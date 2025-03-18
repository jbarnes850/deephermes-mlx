"""
Interactive chat script for DeepHermes-3-Mistral-24B MLX inference.
"""
from typing import List, Dict, Any, Optional
import argparse

from model import load_model
from inference import run_inference


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive chat with DeepHermes-3-Mistral-24B MLX model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="Jarrodbarnes/DeepHermes-3-Mistral-24B-Preview-mlx-fp16",
        help="Model path on Hugging Face Hub"
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
        help="Enable reasoning mode (adds reasoning instruction to system prompt)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run interactive chat."""
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
    
    # Modify system prompt for reasoning if enabled
    system_prompt = args.system_prompt
    if args.reasoning:
        system_prompt += " Please think through this step by step and show your reasoning."
    
    # Initialize chat history
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    print("\n=== DeepHermes-3-Mistral-24B Chat ===")
    print("Type 'exit' to quit, 'clear' to clear history, or 'system <prompt>' to change system prompt.")
    
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
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Format the full conversation for the model
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Simple fallback formatting if no chat template
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Generate response
        print("\nDeepHermes:", end=" ", flush=True)
        response = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=None,  # System prompt already included in messages
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=True,
            verbose=True,
            max_kv_size=args.max_kv_size
        )
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
