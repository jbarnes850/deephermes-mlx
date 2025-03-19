#!/bin/bash
# adaptive_workflow.sh - Unified interface for the DeepHermes Adaptive ML Workflow

# Default parameters
WORKFLOW="general"
PRIORITIZE="balanced"
MAX_MEMORY=80.0
CONFIG_FILE=""
DASHBOARD=false
JSON_OUTPUT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --workflow)
      WORKFLOW="$2"
      shift 2
      ;;
    --prioritize)
      PRIORITIZE="$2"
      shift 2
      ;;
    --max-memory)
      MAX_MEMORY="$2"
      shift 2
      ;;
    --save-config)
      SAVE_CONFIG="$2"
      shift 2
      ;;
    --load-config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    --dashboard)
      DASHBOARD=true
      shift
      ;;
    --help)
      echo "Usage: ./adaptive_workflow.sh [options]"
      echo "Options:"
      echo "  --workflow TYPE      Workflow type: general, content_creation, coding, research"
      echo "  --prioritize TARGET  Performance priority: speed, quality, balanced"
      echo "  --max-memory PCT     Maximum percentage of memory to use (0-100)"
      echo "  --save-config FILE   Save configuration to file"
      echo "  --load-config FILE   Load configuration from file"
      echo "  --json               Output in JSON format"
      echo "  --dashboard          Show hardware and performance dashboard"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

# Build command arguments
ARGS=""

if [ "$DASHBOARD" = true ]; then
  ARGS="$ARGS --dashboard"
fi

if [ "$JSON_OUTPUT" = true ]; then
  ARGS="$ARGS --json"
fi

if [ -n "$CONFIG_FILE" ]; then
  ARGS="$ARGS --load-config $CONFIG_FILE"
else
  ARGS="$ARGS --workflow $WORKFLOW --prioritize $PRIORITIZE --max-memory $MAX_MEMORY"
fi

if [ -n "$SAVE_CONFIG" ]; then
  ARGS="$ARGS --save-config $SAVE_CONFIG"
fi

# Run the adaptive workflow
python -m deephermes.adaptive_workflow.cli $ARGS