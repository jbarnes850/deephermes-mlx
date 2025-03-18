"""
Utility functions for DeepHermes MLX.
"""
from typing import Dict, Optional


def get_reasoning_prompt(depth: str = "deep") -> str:
    """
    Get the reasoning prompt based on the specified depth.
    
    Args:
        depth: Depth of reasoning ('basic', 'deep', or 'expert')
    
    Returns:
        Reasoning prompt to append to system prompt
    """
    # Use the official DeepHermes reasoning prompt
    base_prompt = "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
    
    # Add depth-specific modifications
    if depth == "basic":
        return base_prompt
    elif depth == "deep":
        return base_prompt + " Take your time to explore multiple perspectives and consider different approaches."
    elif depth == "expert":
        return base_prompt + " Thoroughly analyze all aspects of the problem, consider edge cases, evaluate multiple solution paths, and provide a comprehensive explanation of your reasoning process."
    else:
        return base_prompt


def get_default_system_prompt() -> str:
    """
    Get the default system prompt for DeepHermes.
    
    Returns:
        Default system prompt
    """
    return "You are DeepHermes, a helpful AI assistant."
