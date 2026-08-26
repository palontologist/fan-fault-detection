import os
import yaml
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os
import random
import re
from scipy.io import wavfile
from pydub import AudioSegment
import io

from src.stgram_mfn import STgramMFN, compute_anomaly_score


app = FastAPI(title="Fan Fault Detection API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# STgram-MFN detector (lazy loaded)
stgram_detector = None
config = None
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


def load_config(config_path: str = "config.yaml"):
    global config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


def load_stgram_detector(model_path: str, device: torch.device):
    """Load STgram-MFN detector from checkpoint."""
    if config is None:
        load_config()
    
    detector = STgramMFN(num_machine_ids=7).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out batch norm running stats (they have wrong shape in checkpoint)
    state_dict = checkpoint['model_state_dict']
    filtered_state = {k: v for k, v in state_dict.items() 
                      if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
    
    detector.load_state_dict(filtered_state, strict=False)
    
    detector.threshold = checkpoint.get('threshold', 0.5)
    detector.config = config
    detector.device = device
    
    detector.sample_rate = config['data']['audio']['sample_rate']
    detector.n_mels = config['data']['audio']['n_mels']
    detector.n_fft = config['data']['audio']['n_fft']
    detector.hop_length = config['data']['audio']['hop_length']
    detector.duration = config['data']['audio']['duration']
    
    return detector


def get_stgram_detector():
    """Lazy load STgram-MFN detector on first request."""
    global stgram_detector, config
    
    if stgram_detector is not None:
        return stgram_detector
    
    if config is None:
        load_config()
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model_path = Path("checkpoints/stgram_mfn_best.pth")
    
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found")
    
    try:
        global stgram_detector
        stgram_detector = load_stgram_detector(str(model_path), device)
        print(f"Loaded STgram-MFN model (threshold: {stgram_detector.threshold:.4f})")
        return stgram_detector
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")


@app.on_event("startup")
async def startup_event():
    """Fast startup - just verify config loads, lazy load model on first request."""
    if DEMO_MODE:
        print("Running in DEMO MODE - no models loaded")
        return
    
    load_config()
    print("Fast startup complete - model will load on first request")


def preprocess_audio(waveform: torch.Tensor, sr: int, config: dict) -> torch.Tensor:
    if sr != config['data']['audio']['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, config['data']['audio']['sample_rate'])
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    target_length = int(config['data']['audio']['sample_rate'] * config['data']['audio']['duration'])
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['data']['audio']['sample_rate'],
        n_fft=config['data']['audio']['n_fft'],
        hop_length=config['data']['audio']['hop_length'],
        n_mels=config['data']['audio']['n_mels']
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-8)
    
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    return mel_spec.unsqueeze(0)


@app.on_event("startup")
async def startup_event():
    if DEMO_MODE:
        print("Running in DEMO MODE - no models loaded")
        return
    
    load_config()
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model_path = Path("checkpoints/stgram_mfn_best.pth")
    
    if model_path.exists():
        try:
            global stgram_detector
            stgram_detector = load_stgram_detector(str(model_path), device)
            print(f"Loaded STgram-MFN model (threshold: {stgram_detector.threshold:.4f})")
        except Exception as e:
            print(f"Failed to load STgram-MFN model: {e}")
    else:
        print(f"Warning: Model checkpoint not found at checkpoints/stgram_mfn_best.pth")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": stgram_detector is not None,
        "model_type": "STgram-MFN" if stgram_detector else None
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(index_path)


def run_inference(filename: str, contents: bytes, config: dict, demo: bool = False):
    """Run inference with STgram-MFN detector."""
    if demo:
        is_faulty = random.random() > 0.7
        error = random.uniform(0.001, 0.02) if is_faulty else random.uniform(0.0001, 0.005)
        threshold = 0.005
        confidence = min(error / (threshold * 2), 1.0)
        return {
            "filename": filename,
            "is_faulty": is_faulty,
            "reconstruction_error": error,
            "threshold": threshold,
            "confidence": confidence,
            "status": "FAULTY" if is_faulty else "NORMAL",
            "model_used": "demo",
            "demo": True
        }
    
    if config is None:
        load_config()
    
    # Get detector (lazy load)
    detector = get_stgram_detector()
    
    # Load audio with pydub (supports MP3, WAV, FLAC, OGG, etc.)
    try:
        audio = AudioSegment.from_file(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {str(e)}")
    
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Set sample rate
    audio = audio.set_frame_rate(config['data']['audio']['sample_rate'])
    
    # Convert to numpy array
    waveform_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1, 1]
    if audio.sample_width == 2:  # 16-bit
        waveform_np = waveform_np / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        waveform_np = waveform_np / 2147483648.0
    else:
        waveform_np = waveform_np / (2**(audio.sample_width * 8 - 1))
    
    # Pad/trim to target length
    target_length = int(config['data']['audio']['sample_rate'] * config['data']['audio']['duration'])
    if len(waveform_np) > target_length:
        waveform_np = waveform_np[:target_length]
    elif len(waveform_np) < target_length:
        waveform_np = np.pad(waveform_np, (0, target_length - len(waveform_np)))
    
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    
    mel_spec = preprocess_audio(waveform, config['data']['audio']['sample_rate'], config).to(detector.device)
    
    with torch.no_grad():
        # Use STgram-MFN's anomaly scoring
        anomaly_scores, is_faulty = compute_anomaly_score(detector, waveform, detector.device)
        error = float(anomaly_scores[0]) if len(anomaly_scores) > 0 else 0.0
        threshold = detector.threshold
        is_faulty = bool(is_faulty[0]) if hasattr(is_faulty, '__len__') else bool(is_faulty)
        confidence = min(error / (threshold * 2), 1.0) if threshold > 0 else 0.5
    
    return {
        "filename": filename,
        "is_faulty": bool(is_faulty),
        "reconstruction_error": error,
        "threshold": threshold,
        "confidence": confidence,
        "status": "FAULTY" if is_faulty else "NORMAL",
        "model_used": "STgram-MFN"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), demo: bool = Query(False)):
    if not (DEMO_MODE or demo):
        get_stgram_detector()  # Will raise 503 if model not available
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    contents = await file.read()
    
    result = run_inference(file.filename, contents, config, demo)
    return result


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...), demo: bool = Query(False)):
    if not (DEMO_MODE or demo):
        get_stgram_detector()  # Will raise 503 if model not available
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            result = run_inference(file.filename, contents, config, demo)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/threshold")
async def get_threshold():
    if not DEMO_MODE:
        get_stgram_detector()  # Will raise 503 if model not available
    
    if DEMO_MODE:
        return {"threshold": 0.005, "demo": True}
    
    detector = get_stgram_detector()
    return {"threshold": detector.threshold}


@app.get("/health")
async def health_check():
    model_loaded = stgram_detector is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_type": "STgram-MFN" if model_loaded else None
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config['frontend']['host'] if config else "0.0.0.0",
        port=config['frontend']['port'] if config else 8000,
        reload=True
    )