#!/bin/bash

# Gemma3 RKLLM Start Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Gemma3 RKLLM Server...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if RKLLM library exists
if [ ! -f "lib/librkllmrt.so" ]; then
    echo -e "${YELLOW}Warning: RKLLM library not found at lib/librkllmrt.so${NC}"
    echo -e "${YELLOW}Please download librkllmrt.so from Rockchip and place it in the lib/ directory.${NC}"
    echo -e "${YELLOW}Continuing anyway - server will not function until library is installed.${NC}"
fi

# Check if any models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: No models found in models/ directory${NC}"
    echo -e "${YELLOW}Please add your .rkllm model files to continue.${NC}"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start server
echo -e "${GREEN}Server starting on http://localhost:8080${NC}"
python server.py "$@"

