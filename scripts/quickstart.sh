#!/bin/bash
# Quickstart script for DeepHermes MLX

# Parse command line arguments
TEST_MODE=false
TEST_OPTION=""
TEST_DATASET=""
TEST_NUM_EXAMPLES=""
TEST_PROMPT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_MODE=true
      TEST_OPTION="$2"
      if [[ "$TEST_OPTION" == "2" && $# -gt 2 ]]; then
        TEST_DATASET="$3"
        TEST_NUM_EXAMPLES="$4"
        TEST_PROMPT="$5"
        shift 5
      else
        shift 2
      fi
      ;;
    *)
      shift
      ;;
  esac
done

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

# Print success message
echo ""
echo "🎉 Setup complete! DeepHermes-3-Llama-3-8B is ready to use."
echo ""

# Present options to the user
echo "What would you like to do?"
echo "1. Launch chat interface with reasoning"
echo "2. Fine-tune a reasoning model"
echo "3. Use LangChain integration"
echo ""
if [ "$TEST_MODE" = false ]; then
  read -p "Enter your choice (1-3): " choice
else
  choice=$TEST_OPTION
  echo "Selected option: $choice"
fi

case $choice in
    1)
        echo ""
        echo "Starting interactive chat with reasoning mode..."
        echo "Type 'exit' to quit or 'help' to see available commands."
        echo ""
        # Launch the chat interface with reasoning mode
        python chat.py --reasoning
        ;;
    2)
        echo ""
        echo "Starting fine-tuning workflow..."
        echo ""
        
        # Ask for dataset
        echo "Which dataset would you like to use for fine-tuning?"
        echo "1. Alpaca (instruction following)"
        echo "2. Custom dataset (provide path)"
        if [ "$TEST_MODE" = false ]; then
          read -p "Enter your choice (1 or 2): " dataset_choice
        else
          dataset_choice=$TEST_DATASET
          echo "Selected dataset: $dataset_choice"
        fi
        
        if [ "$dataset_choice" = "1" ]; then
            dataset="tatsu-lab/alpaca"
            echo "Using the Alpaca dataset for instruction tuning."
        else
            if [ "$TEST_MODE" = false ]; then
              read -p "Enter the path to your custom dataset: " custom_dataset
              dataset="$custom_dataset"
            else
              dataset=$TEST_DATASET
              echo "Using custom dataset: $dataset"
            fi
        fi
        
        # Ask for number of examples
        if [ "$TEST_MODE" = false ]; then
          read -p "How many examples to use for fine-tuning? (default: 100, use fewer for testing): " num_examples
          num_examples=${num_examples:-100}
        else
          num_examples=$TEST_NUM_EXAMPLES
          echo "Number of examples: $num_examples"
        fi
        
        # Confirm and run
        echo ""
        echo "Ready to fine-tune DeepHermes-3-Llama-3-8B with the following settings:"
        echo "- Dataset: $dataset"
        echo "- Number of examples: $num_examples"
        echo ""
        if [ "$TEST_MODE" = false ]; then
          read -p "Proceed with fine-tuning? (y/n): " confirm
        else
          confirm="y"
          echo "Proceeding with fine-tuning..."
        fi
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            # Run fine-tuning
            ./scripts/finetune_mlx.sh --model "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16" \
                --dataset-name "$dataset" \
                --num-examples "$num_examples" \
                --prepare-data --train
            
            # Ask if user wants to test the model
            echo ""
            if [ "$TEST_MODE" = false ]; then
              read -p "Would you like to test the fine-tuned model with a prompt? (y/n): " test_model
            else
              test_model="y"
              echo "Testing the fine-tuned model..."
            fi
            
            if [ "$test_model" = "y" ] || [ "$test_model" = "Y" ]; then
                if [ "$TEST_MODE" = false ]; then
                  read -p "Enter your prompt: " prompt
                else
                  prompt="$TEST_PROMPT"
                  echo "Using prompt: $prompt"
                fi
                ./scripts/finetune_mlx.sh --model "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16" \
                    --prompt "$prompt" \
                    --generate
            fi
        else
            echo "Fine-tuning cancelled."
        fi
        ;;
    3)
        echo ""
        echo "Starting LangChain integration..."
        echo ""
        
        # Check if model path is provided
        if [ "$TEST_MODE" = false ]; then
            read -p "Do you want to start a new server with a model? (y/n): " start_server
        else
            start_server="n"
            echo "Using existing server..."
        fi
        
        if [ "$start_server" = "y" ] || [ "$start_server" = "Y" ]; then
            if [ "$TEST_MODE" = false ]; then
                read -p "Enter the path to your model: " model_path
            else
                model_path="mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16"
                echo "Using model: $model_path"
            fi
            
            server_args="--model $model_path --start-server"
        else
            if [ "$TEST_MODE" = false ]; then
                read -p "Enter the host (default: 127.0.0.1): " host
                host=${host:-127.0.0.1}
                
                read -p "Enter the port (default: 8080): " port
                port=${port:-8080}
            else
                host="127.0.0.1"
                port="8080"
                echo "Using host: $host, port: $port"
            fi
            
            server_args="--host $host --port $port"
        fi
        
        # Ask for the LangChain mode
        if [ "$TEST_MODE" = false ]; then
            echo ""
            echo "Select LangChain mode:"
            echo "1. Chat interface"
            echo "2. Text generation"
            echo "3. Chain example"
            read -p "Enter your choice (1-3): " lc_mode
        else
            lc_mode="1"
            echo "Using chat interface mode"
        fi
        
        case $lc_mode in
            1)
                echo ""
                echo "Starting chat interface..."
                ./scripts/integrate.sh $server_args chat
                ;;
            2)
                if [ "$TEST_MODE" = false ]; then
                    read -p "Enter your prompt: " prompt
                else
                    prompt="Explain quantum computing in simple terms"
                    echo "Using prompt: $prompt"
                fi
                
                echo ""
                echo "Generating text..."
                ./scripts/integrate.sh $server_args generate --prompt "$prompt"
                ;;
            3)
                if [ "$TEST_MODE" = false ]; then
                    read -p "Enter a topic (default: artificial intelligence): " topic
                    topic=${topic:-"artificial intelligence"}
                    
                    read -p "Enter a concept to explain (default: neural networks): " concept
                    concept=${concept:-"neural networks"}
                else
                    topic="artificial intelligence"
                    concept="neural networks"
                    echo "Using topic: $topic, concept: $concept"
                fi
                
                echo ""
                echo "Running chain example..."
                ./scripts/integrate.sh $server_args chain --topic "$topic" --concept "$concept"
                ;;
            *)
                echo "Invalid choice. Using chat interface."
                ./scripts/integrate.sh $server_args chat
                ;;
        esac
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
