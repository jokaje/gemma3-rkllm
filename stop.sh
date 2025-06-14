#!/bin/bash

# Gemma3 RKLLM Stop Script

echo "Stopping Gemma3 RKLLM Server..."

# Find and kill server processes
pkill -f "python.*server.py" || true

echo "Server stopped."

