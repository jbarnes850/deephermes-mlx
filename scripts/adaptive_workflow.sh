#!/bin/bash
# adaptive_workflow.sh - Unified interface for the DeepHermes Adaptive ML Workflow

# Default parameters
COMMAND="configure"
WORKFLOW="general"
PRIORITIZE="balanced"
MAX_MEMORY=80.0
CONFIG_FILE=""
SAVE_CONFIG=""
JSON_OUTPUT=false
VERBOSE=false
LANGCHAIN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    configure|run|dashboard|list)
      COMMAND="$1"
      shift
      ;;
    --workflow)
      WORKFLOW="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
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
    --verbose)
      VERBOSE=true
      shift
      ;;
    --langchain)
      LANGCHAIN=true
      shift
      ;;
    --help)
      echo "Usage: ./adaptive_workflow.sh [command] [options]"
      echo ""
      echo "Commands:"
      echo "  configure    Configure a workflow without running it"
      echo "  run          Run a workflow"
      echo "  dashboard    Show hardware and performance dashboard"
      echo "  list         List available workflows"
      echo ""
      echo "Options:"
      echo "  --workflow TYPE      Workflow type: general, content_creation, coding, research"
      echo "  --config FILE        Load configuration from file (for run command)"
      echo "  --prioritize TARGET  Performance priority: speed, quality, balanced"
      echo "  --max-memory PCT     Maximum percentage of memory to use (0-100)"
      echo "  --save-config FILE   Save configuration to file"
      echo "  --load-config FILE   Load configuration from file (for configure command)"
      echo "  --json               Output in JSON format"
      echo "  --verbose            Print verbose output (for run command)"
      echo "  --langchain          Use LangChain integration (for run command)"
      echo ""
      echo "Examples:"
      echo "  ./adaptive_workflow.sh configure --workflow coding --prioritize speed"
      echo "  ./adaptive_workflow.sh run --workflow research --langchain"
      echo "  ./adaptive_workflow.sh list --json"
      echo "  ./adaptive_workflow.sh dashboard"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information"
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
ARGS="$COMMAND"

if [ "$COMMAND" = "configure" ]; then
  # Configure command arguments
  ARGS="$ARGS --workflow $WORKFLOW --prioritize $PRIORITIZE --max-memory $MAX_MEMORY"
  
  if [ -n "$CONFIG_FILE" ]; then
    ARGS="$ARGS --load-config $CONFIG_FILE"
  fi
  
  if [ -n "$SAVE_CONFIG" ]; then
    ARGS="$ARGS --save-config $SAVE_CONFIG"
  fi
  
  if [ "$JSON_OUTPUT" = true ]; then
    ARGS="$ARGS --json"
  fi
elif [ "$COMMAND" = "run" ]; then
  # Run command arguments
  ARGS="$ARGS --workflow $WORKFLOW --prioritize $PRIORITIZE --max-memory $MAX_MEMORY"
  
  if [ -n "$CONFIG_FILE" ]; then
    ARGS="$ARGS --config $CONFIG_FILE"
  fi
  
  if [ -n "$SAVE_CONFIG" ]; then
    ARGS="$ARGS --save-config $SAVE_CONFIG"
  fi
  
  if [ "$VERBOSE" = true ]; then
    ARGS="$ARGS --verbose"
  fi
  
  if [ "$LANGCHAIN" = true ]; then
    ARGS="$ARGS --langchain"
  fi
elif [ "$COMMAND" = "list" ]; then
  # List command arguments
  if [ "$JSON_OUTPUT" = true ]; then
    ARGS="$ARGS --json"
  fi
fi

# Run the adaptive workflow
python -m deephermes.adaptive_workflow.cli $ARGS