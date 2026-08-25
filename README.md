# Fan Fault Detection System

AI-powered anomaly detection for industrial fan monitoring using audio analysis. Built with PyTorch, FastAPI, and a modern web frontend.

## Features

- **Autoencoder-based anomaly detection** - CNN autoencoder trained on normal fan sounds
- **Multiple dataset support** - Kaggle, MIMII, DCASE 2022 datasets
- **Real-time inference** - FastAPI backend with web frontend
- **Batch processing** - Analyze multiple files at once
- **Configurable** - YAML-based configuration

## Project Structure

```
fan-fault-detection/
├── config.yaml              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Raw downloaded datasets
│   └── processed/           # Processed data splits
├── src/
│   ├── data_processing.py   # Dataset handling & preprocessing
│   ├── model.py             # Model architectures (CNN/LSTM Autoencoder)
│   ├── train.py             # Training loop
│   ├── inference.py         # CLI inference
│   └── api.py               # FastAPI REST API
├── frontend/
│   └── index.html           # Web interface
├── checkpoints/             # Model checkpoints
├── runs/                    # TensorBoard logs
├── notebooks/               # Jupyter notebooks
└── scripts/                 # Utility scripts
```

## Installation

```bash
cd /home/palontologist/Downloads/dev/fan-fault-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install kaggle CLI (for dataset download)
pip install kaggle
# Configure kaggle.json in ~/.kaggle/
```

## Dataset Setup

Download datasets manually or use the download script:

```bash
# Option 1: Manual download
# 1. Kaggle: https://www.kaggle.com/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan
# 2. MIMII: https://zenodo.org/records/3384388 (select fan machines only)
# 3. DCASE 2022: https://zenodo.org/records/6355122 (select fan machines only)

# Place in data/raw/ with structure:
# data/raw/
#   ├── kaggle_fan/
#   │   ├── normal/
#   │   └── anomaly/
#   ├── mimii_fan/
#   │   ├── normal/
#   │   └── anomaly/
#   └── dcase2022_fan/
#       ├── normal/
#       └── anomaly/

# Option 2: Use download script (requires kaggle API)
python -m src.data_processing
```

## Training

```bash
# Train the model
python main.py train

# Or directly
python -m src.train
```

Training uses:
- CNN Autoencoder architecture
- MSE reconstruction loss
- Cosine annealing LR scheduler
- Early stopping (patience=15)
- TensorBoard logging

## Inference

### CLI
```bash
# Single file
python main.py infer --model checkpoints/best_model.pth --audio test.wav

# Directory (batch)
python main.py infer --model checkpoints/best_model.pth --audio test_folder/
```

### API Server
```bash
# Start API server
python main.py api --model checkpoints/best_model.pth --port 8000

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend
```bash
# Start frontend (in separate terminal)
python main.py frontend --port 3000

# Open http://localhost:3000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Single file prediction |
| `/predict_batch` | POST | Multiple files prediction |
| `/threshold` | GET | Current anomaly threshold |

### Example API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fan_sound.wav"
```

Response:
```json
{
  "filename": "fan_sound.wav",
  "is_faulty": false,
  "reconstruction_error": 0.002341,
  "threshold": 0.005678,
  "confidence": 0.206,
  "status": "NORMAL"
}
```

## Configuration

Edit `config.yaml` to customize:

- **Model architecture**: CNN vs LSTM autoencoder
- **Audio parameters**: Sample rate, mel bands, FFT settings
- **Training hyperparameters**: LR, batch size, epochs, optimizer
- **Data augmentation**: Noise, time stretch, pitch shift
- **Anomaly threshold**: Percentile-based thresholding

## Model Architecture

### CNN Autoencoder (Default)
- **Encoder**: 4 conv blocks (32→64→128→256 channels) + adaptive pooling + FC
- **Latent dim**: 128
- **Decoder**: 4 transposed conv blocks (256→128→64→32→1 channels)
- **Dropout**: 0.3

### LSTM Autoencoder (Alternative)
- **Encoder**: Bidirectional LSTM (2 layers, 256 hidden)
- **Latent dim**: 128
- **Decoder**: LSTM (2 layers, 256 hidden) + FC

## Evaluation Metrics

- **AUC-ROC**: Area under ROC curve
- **pAUC**: Partial AUC (low FPR region)
- **Precision/Recall/F1**: At optimal threshold
- **Threshold**: 95th percentile of normal reconstruction errors

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA recommended)
- 8GB+ RAM (16GB+ for training)
- GPU: NVIDIA with 6GB+ VRAM (recommended)

## License

MIT License

## Acknowledgments

- MIMII Dataset: Purohit et al.
- DCASE 2022 Challenge Task 2
- Kaggle Fan Anomaly Detection Dataset