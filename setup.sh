#!/bin/bash

# Gemma3 RKLLM Setup Script
# Automated installation and configuration for Orange Pi 5 Plus
# Author: AI Assistant
# Version: 1.0.2 (Final)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# The project directory is the same directory where the script is located.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_NAME="gemma3-rkllm"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                  Gemma3 RKLLM Setup                      ║"
    echo "║            Multimodal AI on Orange Pi 5 Plus             ║"
    echo "║                    Version 1.0.2 (Final)                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_platform() {
    print_step "Checking platform compatibility..."
    
    # Check if running on ARM64
    ARCH=$(uname -m)
    if [[ "$ARCH" != "aarch64" ]]; then
        print_error "This script is designed for ARM64 architecture (Orange Pi 5 Plus)"
        print_info "Detected architecture: $ARCH"
        exit 1
    fi
    
    # Check for RK3588 platform
    if [ -f /proc/device-tree/compatible ]; then
        if grep -q "rk3588" /proc/device-tree/compatible; then
            print_info "RK3588 platform detected - Compatible!"
        else
            print_warning "RK3588 not detected, but continuing anyway..."
        fi
    fi
    
    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_info "Operating System: $PRETTY_NAME"
        
        if [[ "$ID" == "ubuntu" ]] || [[ "$ID_LIKE" == *"ubuntu"* ]] || [[ "$ID_LIKE" == *"debian"* ]]; then
            print_info "Debian/Ubuntu-based system detected - Compatible!"
        else
            print_warning "Non-Debian/Ubuntu system detected. Some packages might not install correctly."
        fi
    fi
}

check_requirements() {
    print_step "Checking system requirements..."
    
    # Check available memory
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ "$TOTAL_MEM" -lt 8000 ]; then
        print_warning "Less than 8GB RAM detected ($TOTAL_MEM MB). Performance may be limited."
    else
        print_info "Memory: ${TOTAL_MEM}MB - Sufficient!"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    if [ "$AVAILABLE_GB" -lt 5 ]; then
        print_error "Insufficient disk space. At least 5GB required, found ${AVAILABLE_GB}GB"
        exit 1
    else
        print_info "Disk space: ${AVAILABLE_GB}GB - Sufficient!"
    fi
    
    # Check for sudo privileges
    if ! sudo -n true 2>/dev/null; then
        print_warning "This script requires sudo privileges for some operations. You might be prompted for your password."
    fi
    
    print_info "System requirements check passed!"
}

install_system_dependencies() {
    print_step "Installing system dependencies..."
    
    # Update package list
    print_info "Updating package list..."
    sudo apt-get update -qq
    
    # Install essential packages
    print_info "Installing essential packages..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        htop \
        nano \
        vim
    
    # Install multimedia libraries
    print_info "Installing multimedia libraries..."
    sudo apt-get install -y \
        libopencv-dev \
        python3-opencv \
        libjpeg-dev \
        libpng-dev \
        libwebp-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev
    
    # Install additional Python dependencies
    print_info "Installing additional Python libraries..."
    sudo apt-get install -y \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-pil \
        python3-requests \
        python3-flask
    
    print_info "System dependencies installed successfully!"
}

setup_python_environment() {
    print_step "Setting up Python virtual environment..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install Python dependencies from requirements.txt
    print_info "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r "requirements.txt"
    else
        print_error "requirements.txt not found in ${PROJECT_DIR}!"
        deactivate
        exit 1
    fi

    deactivate
    print_info "Python environment setup completed!"
}

setup_rkllm_library() {
    print_step "Setting up RKLLM library..."
    
    cd "$PROJECT_DIR"
    mkdir -p lib
    
    if [ -f "lib/librkllmrt.so" ]; then
        print_info "RKLLM library already exists"
        return
    fi
    
    print_info "RKLLM library not found. Please follow these steps:"
    echo ""
    echo "1. Download the RKLLM library from Rockchip:"
    echo "   - Visit: https://github.com/airockchip/rknn-llm"
    echo "   - Download the appropriate librkllmrt.so for RK3588"
    echo ""
    echo "2. Copy the library to: $PROJECT_DIR/lib/librkllmrt.so"
    echo ""
    echo "3. Re-run this setup script"
    echo ""
    
    read -p "Do you want to continue without the RKLLM library? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup paused. Please install the RKLLM library and re-run."
        exit 0
    fi
    
    print_warning "Continuing without RKLLM library. The server will not function until the library is installed."
}

setup_directories() {
    print_step "Setting up project directories..."
    
    cd "$PROJECT_DIR"
    
    mkdir -p models logs config scripts docs tests
    chmod 755 models logs config scripts docs tests
    
    if [ ! -d "models/example" ]; then
        mkdir -p models/example
        cat > models/example/Modelfile << EOF
FROM="gemma3-4b.rkllm"
HUGGINGFACE_PATH="google/gemma-2-2b-it"
SYSTEM="You are a helpful AI assistant with vision capabilities."
TEMPERATURE=0.7
TOKENIZER="google/gemma-2-2b-it"
EOF
        print_info "Created example model configuration"
    fi
    
    print_info "Project directories setup completed!"
}

configure_npu_optimization() {
    print_step "Configuring NPU optimization..."
    
    if [ -d "/sys/class/devfreq" ]; then
        NPU_DEVFREQ=$(find /sys/class/devfreq -name "*npu*" | head -1)
        if [ -n "$NPU_DEVFREQ" ]; then
            print_info "NPU devfreq found: $NPU_DEVFREQ"
            
            if [ -w "$NPU_DEVFREQ/governor" ]; then
                echo "performance" | sudo tee "$NPU_DEVFREQ/governor" > /dev/null
                print_info "NPU governor set to performance mode"
            else
                print_warning "Cannot set NPU governor (insufficient permissions or not running with sudo)"
            fi
        else
            print_warning "NPU devfreq not found"
        fi
    else
        print_warning "devfreq subsystem not found"
    fi
    
    print_info "Setting CPU governor to performance mode..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -w "$cpu" ]; then
            echo "performance" | sudo tee "$cpu" > /dev/null
        fi
    done
    
    print_info "Disabling swap for optimal performance..."
    sudo swapoff -a 2>/dev/null || true
    
    print_info "NPU optimization configuration completed!"
}

create_service_files() {
    print_step "Creating systemd service files..."
    
    cat > /tmp/gemma3-rkllm.service << EOF
[Unit]
Description=Gemma3 RKLLM Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/server.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/gemma3-rkllm.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    print_info "Systemd service created. Use 'sudo systemctl enable gemma3-rkllm' to enable auto-start"
}

create_startup_scripts() {
    print_step "Creating startup scripts..."
    
    cd "$PROJECT_DIR"
    
    # Create/Verify start.sh, stop.sh, client.py from the uploaded versions
    # This assumes the files are present and correct, and just ensures they are executable
    if [ -f "start.sh" ] && [ -f "stop.sh" ] && [ -f "client.py" ]; then
        chmod +x "start.sh" "stop.sh" "client.py"
        print_info "Startup scripts are present and executable."
    else
        print_error "One or more startup scripts (start.sh, stop.sh, client.py) are missing!"
    fi
}

run_tests() {
    print_step "Running basic tests..."
    
    cd "$PROJECT_DIR"
    source venv/bin/activate
    
    # --- KORREKTUR 3: Python-Import-Test repariert ---
    print_info "Testing Python imports..."
    # We run from the project root, so 'src' is treated as a package.
    # We use absolute imports like 'from src.module'
    python -c "
import sys
import os
try:
    from src.config_manager import ConfigManager
    # You can add more imports here to test other modules
    # from src.image_processor import ImageProcessor
    print('✓ Core modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    print(f'   This usually means a problem with relative imports inside the src package, or a missing __init__.py file.')
    print(f'   Current working directory: {os.getcwd()}')
    print(f'   Sys path: {sys.path}')
    deactivate
    exit(1)
"
    
    print_info "Testing configuration loading..."
    python -c "
import sys
import os
from src.config_manager import ConfigManager
try:
    config_manager = ConfigManager()
    # The config file path is relative to the project root
    config = config_manager.load_config('config/default.ini')
    print('✓ Configuration loads successfully')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    deactivate
    exit(1)
"
    deactivate
    print_info "Basic tests completed successfully!"
}

print_completion_info() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete!                       ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Install your Gemma3 RKLLM model (if you have one):"
    echo "   - Place your .rkllm file in: $PROJECT_DIR/models/your-model-name/"
    echo "   - Create or update the Modelfile in the same directory"
    echo ""
    echo "2. Install the RKLLM library (if not done already):"
    echo "   - Download librkllmrt.so from Rockchip"
    echo "   - Place it in: $PROJECT_DIR/lib/librkllmrt.so"
    echo ""
    echo "3. Start the server:"
    echo "   cd $PROJECT_DIR && ./start.sh"
    echo ""
    echo "4. Test the server from another terminal:"
    echo "   cd $PROJECT_DIR && ./client.py --prompt \"Hello, how are you?\""
    echo ""
    echo "Server will be available at: http://localhost:8080"
}

# Main execution
main() {
    print_header
    
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root. Use a regular user with sudo privileges."
        exit 1
    fi
    
    check_platform
    check_requirements
    install_system_dependencies
    setup_python_environment
    setup_rkllm_library
    setup_directories
    configure_npu_optimization
    create_service_files
    create_startup_scripts
    run_tests
    print_completion_info
}

# Run main function
main "$@"

