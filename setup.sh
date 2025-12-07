#!/bin/bash
# Automated setup script for FL Smart Home Privacy Project

echo "=================================="
echo "FL Smart Home Privacy Setup"
echo "=================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python 3 found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo -e "\n[4/5] Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create results directory
echo -e "\n[5/5] Creating directories..."
mkdir -p results data
echo "✓ Directories created"

echo -e "\n=================================="
echo "Setup Complete!"
echo "=================================="
echo -e "\nTo run experiments:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run experiments: python run_all_experiments.py"
echo -e "\nFor quick test:"
echo "  python data/generate_data.py"
echo "=================================="
