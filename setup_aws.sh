#!/bin/bash

# Setup script for AWS Ubuntu instance
set -e

echo "🚀 Starting Setup..."

# 1. System Updates
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv screen

# 2. Python Environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

# 3. Dependencies
echo "📚 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verification
echo "🔍 Verifying setup..."
python3 -c "import py_clob_client; print('✅ py_clob_client installed')"
python3 -c "import requests; print('✅ requests installed')"

echo ""
echo "🎉 Setup Complete!"
echo "To run the bot:"
echo "  source venv/bin/activate"
echo "  python3 main_hopper.py"
echo ""
echo "Tip: Use 'screen' to keep it running in the background."

