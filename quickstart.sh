#!/bin/bash
# Quick start script for Fan Fault Detection

set -e

PROJECT_DIR="/home/palontologist/Downloads/dev/fan-fault-detection"

echo "=========================================="
echo "Fan Fault Detection - Quick Start"
echo "=========================================="

cd "$PROJECT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed checkpoints runs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. DOWNLOAD DATASETS:"
echo "   - Kaggle: https://www.kaggle.com/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan"
echo "   - MIMII: https://zenodo.org/records/3384388 (select fan only)"
echo "   - DCASE 2022: https://zenodo.org/records/6355122 (select fan only)"
echo ""
echo "2. ORGANIZE DATA in data/raw/:"
echo "   data/raw/kaggle_fan/normal/"
echo "   data/raw/kaggle_fan/anomaly/"
echo "   data/raw/mimii_fan/normal/"
echo "   data/raw/mimii_fan/anomaly/"
echo "   data/raw/dcase2022_fan/normal/"
echo "   data/raw/dcase2022_fan/anomaly/"
echo ""
echo "3. PREPARE DATA:"
echo "   python scripts/prepare_data.py"
echo ""
echo "4. TRAIN MODEL:"
echo "   python main.py train"
echo ""
echo "5. RUN INFERENCE:"
echo "   python main.py infer --model checkpoints/best_model.pth --audio test.wav"
echo ""
echo "6. START API SERVER:"
echo "   python main.py api --model checkpoints/best_model.pth"
echo ""
echo "7. START FRONTEND (in new terminal):"
echo "   python main.py frontend"
echo "   Then open http://localhost:3000"
echo ""
echo "=========================================="