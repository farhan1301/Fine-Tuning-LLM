#!/bin/bash

# Setup script for LLM Fine-tuning Project
# This script automates the conda environment setup

set -e  # Exit on error

echo "=================================================="
echo "LLM Fine-tuning Project - Setup Script"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is installed
echo -e "${BLUE}Checking for conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found!${NC}"
    echo "Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓ Conda found${NC}"
echo ""

# Get project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo -e "${BLUE}Project directory: ${PROJECT_DIR}${NC}"
echo ""

# Check if environment.yml exists
if [ ! -f "${PROJECT_DIR}/environment.yml" ]; then
    echo -e "${RED}✗ environment.yml not found!${NC}"
    exit 1
fi

# Check if environment already exists
echo -e "${BLUE}Checking if environment 'llm-finetuning' already exists...${NC}"
if conda env list | grep -q "llm-finetuning"; then
    echo -e "${YELLOW}⚠ Environment 'llm-finetuning' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing environment...${NC}"
        conda env remove -n llm-finetuning -y
        echo -e "${GREEN}✓ Environment removed${NC}"
    else
        echo -e "${YELLOW}Using existing environment${NC}"
        conda activate llm-finetuning
        echo -e "${GREEN}✓ Environment activated${NC}"
        exit 0
    fi
fi
echo ""

# Create conda environment
echo -e "${BLUE}Creating conda environment 'llm-finetuning'...${NC}"
echo "This may take 5-10 minutes..."
echo ""
conda env create -f "${PROJECT_DIR}/environment.yml"
echo ""
echo -e "${GREEN}✓ Conda environment created${NC}"
echo ""

# Activate environment
echo -e "${BLUE}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-finetuning
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python -c "
import sys
try:
    import torch
    import transformers
    import peft
    import trl
    import pydantic
    import gradio
    print('✓ All core packages imported successfully!')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# Register Jupyter kernel
echo -e "${BLUE}Registering Jupyter kernel...${NC}"
python -m ipykernel install --user --name=llm-finetuning --display-name="Python (LLM Fine-tuning)" --force
echo -e "${GREEN}✓ Jupyter kernel registered${NC}"
echo ""

# Print summary
echo "=================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   ${BLUE}conda activate llm-finetuning${NC}"
echo ""
echo "2. Set up API keys (required):"
echo "   ${BLUE}export HUGGINGFACE_TOKEN='your_token_here'${NC}"
echo "   ${BLUE}export ANTHROPIC_API_KEY='your_key_here'${NC}  (or OpenAI)"
echo ""
echo "3. Login to Hugging Face:"
echo "   ${BLUE}huggingface-cli login${NC}"
echo ""
echo "4. Launch Jupyter:"
echo "   ${BLUE}jupyter notebook${NC}"
echo ""
echo "5. Open and run notebooks in order:"
echo "   - notebooks/01_setup.ipynb"
echo "   - notebooks/02_data_generation.ipynb"
echo "   - notebooks/03_data_preparation.ipynb"
echo "   - notebooks/04_training.ipynb"
echo "   - notebooks/05_evaluation.ipynb"
echo "   - notebooks/06_inference.ipynb"
echo "   - notebooks/07_demo.ipynb"
echo ""
echo "For detailed instructions, see SETUP_GUIDE.md"
echo ""
echo -e "${GREEN}Happy fine-tuning! 🚀${NC}"
echo ""
