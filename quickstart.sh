#!/bin/bash
# Quickstart script for DeepHermes MLX

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download the model
echo ""
echo "Downloading the DeepHermes-3-Llama-3-8B model (this may take a few minutes)..."
python -c "from transformers import AutoTokenizer; from mlx_lm import load; model, tokenizer = load('mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16'); print('Model downloaded successfully!')"

# Print success message and instructions
echo ""
echo "🎉 Setup complete! DeepHermes-3-Llama-3-8B is ready to use."
echo ""
echo "Quick commands:"
echo "---------------"
echo "Interactive chat:              python chat.py --reasoning"
echo "Memory-optimized chat:         python chat.py --reasoning --quantize 4bit"
echo "Single prompt:                 python main.py --reasoning --prompt \"Your prompt here\""
echo ""
echo "For more examples, see examples.md"
echo ""
