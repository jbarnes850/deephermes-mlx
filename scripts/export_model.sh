#!/bin/bash
# Export a DeepHermes model for deployment

# Default parameters
MODEL=""
BASE_MODEL=""
OUTPUT_DIR="./exported_model"
QUANTIZE="none"
FORMAT="mlx"
VALIDATE=false
MODEL_NAME=""
DESCRIPTION=""
AUTHOR=""
TAGS=""
DEMO=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --quantize)
      QUANTIZE="$2"
      shift 2
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --validate)
      VALIDATE=true
      shift
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --description)
      DESCRIPTION="$2"
      shift 2
      ;;
    --author)
      AUTHOR="$2"
      shift 2
      ;;
    --tags)
      TAGS="$2"
      shift 2
      ;;
    --demo)
      DEMO=true
      shift
      ;;
    --help)
      echo "Usage: ./export_model.sh [options]"
      echo "Options:"
      echo "  --model MODEL       Model ID or path (required unless --demo is used)"
      echo "  --base-model MODEL  Base model path (required for LoRA adapters)"
      echo "  --output DIR        Output directory (default: ./exported_model)"
      echo "  --quantize TYPE     Quantization type: int8, int4, fp16, none (default: none)"
      echo "  --format FORMAT     Output format: mlx (default: mlx)"
      echo "  --validate          Validate the exported model"
      echo "  --model-name NAME   Name for the exported model"
      echo "  --description DESC  Description for the exported model"
      echo "  --author AUTHOR     Author of the exported model"
      echo "  --tags TAGS         Tags for the exported model (space-separated)"
      echo "  --demo              Run in demo mode with a minimal test model"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if model is provided (unless in demo mode)
if [ -z "$MODEL" ] && [ "$DEMO" = false ]; then
  echo "Error: Model is required. Use --model to specify or use --demo for demonstration mode."
  exit 1
fi

# Export the model
echo "Exporting model..."

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Check if this is a LoRA adapter
if [ -f "$MODEL/adapter_config.json" ] || [ -f "$MODEL/adapters.safetensors" ] || [ -f "$MODEL/0000100_adapters.safetensors" ] || [ -f "$MODEL/adapter_model.safetensors" ]; then
  echo "Detected LoRA adapter..."
  
  # Check if base model is provided
  if [ -z "$BASE_MODEL" ]; then
    echo "Error: Base model is required for LoRA adapters. Use --base-model to specify."
    exit 1
  fi
fi

# Build command with all arguments
CMD="python -m deephermes.cli.export_cli"

# Add model argument if not in demo mode or if model is provided
if [ -n "$MODEL" ]; then
  CMD="$CMD --model \"$MODEL\""
elif [ "$DEMO" = true ]; then
  # Use a placeholder model path in demo mode
  CMD="$CMD --model \"demo_model\" --demo"
fi

CMD="$CMD --output \"$OUTPUT_DIR\" --quantize \"$QUANTIZE\" --format \"$FORMAT\""

# Add optional arguments if provided
if [ -n "$BASE_MODEL" ]; then
  CMD="$CMD --base-model \"$BASE_MODEL\""
fi

if [ "$VALIDATE" = true ]; then
  CMD="$CMD --validate"
fi

if [ "$DEMO" = true ]; then
  CMD="$CMD --demo"
fi

if [ -n "$MODEL_NAME" ]; then
  CMD="$CMD --model-name \"$MODEL_NAME\""
fi

if [ -n "$DESCRIPTION" ]; then
  CMD="$CMD --description \"$DESCRIPTION\""
fi

if [ -n "$AUTHOR" ]; then
  CMD="$CMD --author \"$AUTHOR\""
fi

if [ -n "$TAGS" ]; then
  CMD="$CMD --tags $TAGS"
fi

# Execute the command
echo "Running: $CMD"
eval $CMD

echo ""
echo "Model successfully exported to $OUTPUT_DIR"

# Test the exported model
echo "Testing exported model..."
python scripts/test_exported_model.py --model "$OUTPUT_DIR"

echo "Export complete. Model saved to $OUTPUT_DIR"
