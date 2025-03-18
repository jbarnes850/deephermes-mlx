"""
Inference module for DeepHermes-3-Mistral-24B MLX inference.
"""
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizer
from mlx_lm import generate, stream_generate
from mlx_lm.sample_utils import make_sampler


def format_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    as_chat: bool = True
) -> str:
    """
    Format the prompt for the model.
    
    Args:
        tokenizer: The tokenizer to use
        prompt: The user prompt (optional if messages is provided)
        messages: List of message dictionaries with role and content (optional if prompt is provided)
        system_prompt: Optional system prompt (used only if messages is None)
        as_chat: Whether to format as a chat message
    
    Returns:
        Formatted prompt string
    """
    if not as_chat:
        return prompt if prompt is not None else ""
    
    if messages is not None:
        # Use provided messages
        chat_messages = messages
    else:
        # Create messages from prompt and system_prompt
        chat_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # Add user message if prompt is provided
        if prompt:
            chat_messages.append({"role": "user", "content": prompt})
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt
    
    # Fallback formatting if no chat template
    if prompt is not None:
        return prompt
    elif messages is not None and len(messages) > 0:
        # Simple fallback: concatenate all messages
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    else:
        return ""


def run_inference(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
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
        prompt: The user prompt (optional if messages is provided)
        messages: List of message dictionaries with role and content (optional if prompt is provided)
        system_prompt: Optional system prompt (used only if messages is None)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the output
        verbose: Whether to print verbose output
        **kwargs: Additional arguments to pass to generate
    
    Returns:
        Generated text
    """
    # Ensure at least one of prompt or messages is provided
    if prompt is None and messages is None:
        raise ValueError("Either prompt or messages must be provided")
    
    # Format the prompt
    formatted_prompt = format_prompt(tokenizer, prompt, messages, system_prompt)
    
    # Create a sampler function with the temperature and top_p parameters
    sampler = make_sampler(temp=temperature, top_p=top_p)
    
    # Filter out parameters that are handled separately
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['temperature', 'top_p', 'max_tokens']}
    
    # Generate text
    if stream:
        # Stream the output
        full_response = ""
        for response in stream_generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **filtered_kwargs
        ):
            if verbose:
                print(response.text, end="", flush=True)
            full_response += response.text
        
        if verbose:
            print()  # Add newline at the end
        
        return full_response
    else:
        # Generate all at once using the sampler
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=verbose,
            **filtered_kwargs
        )
        return response
