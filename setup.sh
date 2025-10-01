#!/bin/bash
# setup.sh - Installation and setup script

echo "=========================================="
echo "Scene Completion Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (modify based on your CUDA version)
echo ""
echo "Installing PyTorch..."
echo "Select your CUDA version:"
echo "1. CUDA 11.8"
echo "2. CUDA 12.1"
echo "3. CPU only"
read -p "Enter choice (1-3): " cuda_choice

case $cuda_choice in
    1)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Invalid choice, installing CPU version"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data
mkdir -p outputs
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

# Download sample data (optional)
read -p "Download sample data for testing? (y/n): " download_data
if [ "$download_data" = "y" ]; then
    echo "Downloading sample data..."
    # Add download commands here
    echo "Sample data would be downloaded here"
fi

# Test installation
echo ""
echo "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run quick test:"
echo "  python quick_start.py --mode train --epochs 5"
echo ""
echo "To train on your data:"
echo "  python train.py --data_type video --data_dir ./data --epochs 100"