# DeepHermes Examples

This document provides example prompts and use cases to help you get started with the DeepHermes-3-Llama-3-8B model.

## Reasoning Examples

The DeepHermes model excels at step-by-step reasoning. Here are some examples to try:

### Mathematical Reasoning

```bash
python main.py --reasoning --prompt "Calculate the compound interest on $1000 invested for 5 years at an annual rate of 8% compounded quarterly."
```

### Logical Reasoning

```bash
python main.py --reasoning --prompt "If all A are B, and some B are C, can we conclude that some A are C? Explain your reasoning."
```

### Creative Problem Solving

```bash
python main.py --reasoning --prompt "Design a system to reduce traffic congestion in a major city. Consider different stakeholders and potential unintended consequences."
```

## Coding Examples

DeepHermes can help with coding tasks:

```bash
python main.py --reasoning --prompt "Write a Python function that takes a list of integers and returns the two numbers that add up to a specific target."
```

## Chat Commands Reference

During an interactive chat session:

- `exit`: Quit the chat session
- `clear`: Clear chat history
- `system <prompt>`: Change the system prompt
- `reasoning on`: Enable reasoning mode with <think> tags
- `reasoning off`: Disable reasoning mode

## One-Liner Examples

Quick commands to run specific tasks:

```bash
# Creative writing with reasoning
python main.py --reasoning --prompt "Write a short story about a robot discovering emotions"

# Problem solving
python main.py --reasoning --prompt "What are three approaches to optimize a slow database query?"

# Interactive chat with memory optimization
python chat.py --reasoning --quantize 4bit

# Maximum performance (full precision)
python chat.py --reasoning
```

## Tips for Best Results

1. For complex reasoning tasks, always use the `--reasoning` flag
2. For longer conversations, use the chat interface instead of main.py
3. If you're experiencing memory issues, try `--quantize 4bit --lazy-load`
