#!/usr/bin/env python3
"""
Test script for the DeepHermes MLX model with the workflow runner.
"""
import sys
from deephermes.adaptive_workflow.workflow_runner import WorkflowRunner

def main():
    # Create a workflow runner with the research workflow
    runner = WorkflowRunner(
        workflow_type="research",
        prioritize_quality=True,
        verbose=True
    )
    
    # Initialize the model
    print("Initializing model...")
    runner.initialize()
    
    # Test prompt
    prompt = "Explain the potential impact of quantum computing on modern cryptography and data security. Include both challenges and opportunities."
    
    # Generate response
    print("\nGenerating response to prompt:")
    print(f"Prompt: {prompt}\n")
    print("-" * 80)
    
    response = runner.generate_response(prompt)
    
    print("-" * 80)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
