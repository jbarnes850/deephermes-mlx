"""
Workflow templates for the Adaptive ML Workflow.

This module provides specialized workflow templates for different use cases,
each optimized for specific tasks and hardware capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import json
from dataclasses import dataclass
from .hardware_profiles import AppleSiliconProfile
from ..core.utils import get_reasoning_prompt, get_default_system_prompt


@dataclass
class WorkflowTemplate:
    """Template for an adaptive ML workflow."""
    name: str
    description: str
    system_prompt: str
    reasoning_enabled: bool
    reasoning_depth: str
    model_config_adjustments: Dict[str, Any]
    fine_tuning_config_adjustments: Dict[str, Any]
    serving_config_adjustments: Dict[str, Any]
    integration_config_adjustments: Dict[str, Any]
    
    def get_full_system_prompt(self) -> str:
        """Get the full system prompt including reasoning if enabled."""
        if self.reasoning_enabled:
            return self.system_prompt + get_reasoning_prompt(self.reasoning_depth)
        return self.system_prompt


# General purpose workflow template
GENERAL_WORKFLOW = WorkflowTemplate(
    name="general",
    description="Balanced configuration for general-purpose use",
    system_prompt=get_default_system_prompt(),
    reasoning_enabled=True,
    reasoning_depth="deep",
    model_config_adjustments={
        # No specific adjustments needed
    },
    fine_tuning_config_adjustments={
        # Balanced fine-tuning configuration
    },
    serving_config_adjustments={
        "temperature": 0.7,
        "top_p": 0.9,
    },
    integration_config_adjustments={
        # No specific adjustments needed
    }
)


# Content creation workflow template
CONTENT_CREATION_WORKFLOW = WorkflowTemplate(
    name="content_creation",
    description="Optimized for generating creative content",
    system_prompt=(
        "You are DeepHermes, a creative AI assistant specialized in content creation. "
        "You excel at generating engaging, original, and high-quality content across various formats "
        "including blog posts, stories, marketing copy, social media content, and more. "
        "Your writing is vivid, engaging, and tailored to the intended audience. "
        "You can adapt your tone and style to match different creative contexts."
    ),
    reasoning_enabled=False,  # Creative tasks often benefit from more fluid thinking
    reasoning_depth="basic",
    model_config_adjustments={
        # Prefer larger models for creativity if available
    },
    fine_tuning_config_adjustments={
        # Adjust for creative content generation
    },
    serving_config_adjustments={
        "temperature": 0.85,  # Higher temperature for more creative outputs
        "top_p": 0.92,
        "top_k": 60,  # More diversity in token selection
    },
    integration_config_adjustments={
        # No specific adjustments needed
    }
)


# Coding workflow template
CODING_WORKFLOW = WorkflowTemplate(
    name="coding",
    description="Specialized for code generation and completion",
    system_prompt=(
        "You are DeepHermes, an expert AI programming assistant. "
        "You specialize in generating clean, efficient, and well-documented code. "
        "You follow best practices for the language you're working with, including "
        "proper error handling, commenting, and adherence to style guides. "
        "You can explain your code clearly and provide context for your implementation decisions. "
        "When generating code, you ensure it's secure, optimized, and maintainable."
    ),
    reasoning_enabled=True,
    reasoning_depth="expert",  # Deep reasoning for complex coding tasks
    model_config_adjustments={
        # Prefer models with strong coding capabilities
    },
    fine_tuning_config_adjustments={
        # Adjust for code generation
    },
    serving_config_adjustments={
        "temperature": 0.4,  # Lower temperature for more precise code generation
        "top_p": 0.95,
        "max_tokens_per_request": 8192,  # Longer context for code generation
    },
    integration_config_adjustments={
        "chunk_size": 1024,  # Larger chunks for code blocks
    }
)


# Research workflow template
RESEARCH_WORKFLOW = WorkflowTemplate(
    name="research",
    description="Focused on high-quality, in-depth responses",
    system_prompt=(
        "You are DeepHermes, an advanced AI research assistant. "
        "You excel at in-depth analysis, critical thinking, and comprehensive research. "
        "You can analyze complex topics from multiple perspectives, evaluate evidence, "
        "identify patterns, and synthesize information into clear, well-structured responses. "
        "You cite sources when appropriate and acknowledge limitations in your knowledge. "
        "Your goal is to provide nuanced, accurate, and thorough responses to research queries."
    ),
    reasoning_enabled=True,
    reasoning_depth="expert",  # Maximum reasoning for research tasks
    model_config_adjustments={
        # Prefer largest available models with no quantization for research
    },
    fine_tuning_config_adjustments={
        # Adjust for research capabilities
    },
    serving_config_adjustments={
        "temperature": 0.5,  # Balanced temperature for factual yet nuanced responses
        "top_p": 0.85,
        "max_tokens_per_request": 8192,  # Longer context for in-depth research
        "context_window": 16384,  # Maximize context window if hardware supports it
    },
    integration_config_adjustments={
        "use_memory_efficient_attention": False,  # Prioritize quality over memory efficiency
    }
)


# Dictionary of all available workflow templates
WORKFLOW_TEMPLATES = {
    "general": GENERAL_WORKFLOW,
    "content_creation": CONTENT_CREATION_WORKFLOW,
    "coding": CODING_WORKFLOW,
    "research": RESEARCH_WORKFLOW,
}


def get_workflow_template(workflow_type: str) -> WorkflowTemplate:
    """
    Get a workflow template by name.
    
    Args:
        workflow_type: Name of the workflow template
        
    Returns:
        The workflow template
        
    Raises:
        ValueError: If the workflow template does not exist
    """
    if workflow_type not in WORKFLOW_TEMPLATES:
        raise ValueError(f"Unknown workflow type: {workflow_type}. Available types: {', '.join(WORKFLOW_TEMPLATES.keys())}")
    
    return WORKFLOW_TEMPLATES[workflow_type]


def apply_workflow_template(config: Dict[str, Any], template: WorkflowTemplate) -> Dict[str, Any]:
    """
    Apply a workflow template to a configuration.
    
    Args:
        config: Configuration to modify
        template: Workflow template to apply
        
    Returns:
        Modified configuration
    """
    # Create a copy of the configuration
    modified_config = config.copy()
    
    # Apply model config adjustments
    if "model_config" in modified_config and template.model_config_adjustments:
        for key, value in template.model_config_adjustments.items():
            modified_config["model_config"][key] = value
    
    # Apply fine-tuning config adjustments
    if "fine_tuning_config" in modified_config and template.fine_tuning_config_adjustments:
        for key, value in template.fine_tuning_config_adjustments.items():
            modified_config["fine_tuning_config"][key] = value
    
    # Apply serving config adjustments
    if "serving_config" in modified_config and template.serving_config_adjustments:
        for key, value in template.serving_config_adjustments.items():
            modified_config["serving_config"][key] = value
    
    # Apply integration config adjustments
    if "integration_config" in modified_config and template.integration_config_adjustments:
        for key, value in template.integration_config_adjustments.items():
            modified_config["integration_config"][key] = value
    
    # Update workflow type
    modified_config["workflow_type"] = template.name
    
    return modified_config
