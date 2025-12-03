#!/bin/bash
# Universal Quick Start for Hybrid Quantum GAN Drug Discovery
# Works on macOS, Linux, and Windows (Git Bash/WSL)

set -e

echo "🚀 Hybrid Quantum GAN Drug Discovery - Quick Start"
echo "=================================================="
echo ""

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.10+ and try again."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PY_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "❌ Could not find activation script"
    exit 1
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p data/raw data/processed data/interim
mkdir -p experiments/checkpoints experiments/final_validation
mkdir -p logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Launch UI:    streamlit run src/ui/streamlit_app.py"
echo "   2. Or run:       ./QUICK_START.sh --ui"
echo ""

# If --ui flag is passed, launch the UI
if [ "$1" = "--ui" ]; then
    echo "🌐 Launching Streamlit UI..."
    streamlit run src/ui/streamlit_app.py --server.port=8501 --server.headless=true
fi
