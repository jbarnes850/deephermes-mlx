"""
Inference module for DeepHermes-3-Mistral-24B MLX inference.
"""
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizer
from mlx_lm import generate, stream_generate


def format_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    as_chat: bool = True
) -> str:
    """
    Format the prompt for the model.
    
    Args:
        tokenizer: The tokenizer to use
        prompt: The user prompt
        system_prompt: Optional system prompt
        as_chat: Whether to format as a chat message
    
    Returns:
        Formatted prompt string
    """
    if not as_chat:
        return prompt
    
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt
    
    # Fallback formatting if no chat template
    return prompt


def run_inference(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = True,
    verbose: bool = True,
    **kwargs
) -> str:
    """
    Run inference on the model.
    
    Args:
        model: The MLX model
        tokenizer: The tokenizer
        prompt: The user prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the output
        verbose: Whether to print verbose output
        **kwargs: Additional arguments to pass to generate
    
    Returns:
        Generated text
    """
    # Format the prompt
    formatted_prompt = format_prompt(tokenizer, prompt, system_prompt)
    
    # Generate text
    if stream:
        # Stream the output
        full_response = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        ):
            if verbose:
                print(response.text, end="", flush=True)
            full_response += response.text
        
        if verbose:
            print()  # Add newline at the end
        
        return full_response
    else:
        # Generate all at once
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            **kwargs
        )
        return response
