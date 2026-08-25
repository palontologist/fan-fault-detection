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
from scipy.io import wavfile

from src.model import load_model, CNNAutoencoder


app = FastAPI(title="Fan Fault Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

detector = None
config = None
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


def load_config(config_path: str = "config.yaml"):
    global config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


def initialize_detector(model_path: str):
    global detector, config
    if config is None:
        load_config()
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    detector = load_model(model_path, config, device)
    
    checkpoint = torch.load(model_path, map_location=device)
    detector.threshold = checkpoint.get('threshold', 0.5)
    detector.config = config
    detector.device = device
    
    detector.sample_rate = config['data']['audio']['sample_rate']
    detector.n_mels = config['data']['audio']['n_mels']
    detector.n_fft = config['data']['audio']['n_fft']
    detector.hop_length = config['data']['audio']['hop_length']
    detector.duration = config['data']['audio']['duration']
    
    return detector


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
    model_path = os.environ.get("MODEL_PATH", "checkpoints/best_model.pth")
    if os.path.exists(model_path):
        initialize_detector(model_path)
        print(f"Model loaded from {model_path}")
    elif DEMO_MODE:
        print("Running in DEMO MODE - no model loaded, returning mock predictions")
    else:
        print(f"Warning: Model not found at {model_path}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(index_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...), demo: bool = Query(False)):
    if detector is None and not (DEMO_MODE or demo):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if DEMO_MODE or demo:
        await file.read()
        is_faulty = random.random() > 0.7
        error = random.uniform(0.001, 0.02) if is_faulty else random.uniform(0.0001, 0.005)
        threshold = 0.005
        confidence = min(error / (threshold * 2), 1.0)
        
        return {
            "filename": file.filename,
            "is_faulty": is_faulty,
            "reconstruction_error": error,
            "threshold": threshold,
            "confidence": confidence,
            "status": "FAULTY" if is_faulty else "NORMAL",
            "demo": True
        }
    
    if config is None:
        load_config()
    
    allowed_extensions = config['frontend']['allowed_extensions']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {allowed_extensions}"
        )
    
    try:
        contents = await file.read()
        
        # Load audio with scipy
        sr, waveform_np = wavfile.read(io.BytesIO(contents))
        
        # Convert to float32 normalized
        if waveform_np.dtype == np.int16:
            waveform_np = waveform_np.astype(np.float32) / 32768.0
        elif waveform_np.dtype == np.int32:
            waveform_np = waveform_np.astype(np.float32) / 2147483648.0
        else:
            waveform_np = waveform_np.astype(np.float32)
        
        # Handle stereo
        if len(waveform_np.shape) > 1:
            waveform_np = waveform_np.mean(axis=1)
        
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        mel_spec = preprocess_audio(waveform, sr, config).to(detector.device)
        
        with torch.no_grad():
            reconstructed, latent = detector(mel_spec)
            error = detector.get_reconstruction_error(mel_spec).item()
            
            is_faulty = error > detector.threshold
            confidence = min(error / (detector.threshold * 2), 1.0) if detector.threshold > 0 else 0.5
        
        return {
            "filename": file.filename,
            "is_faulty": bool(is_faulty),
            "reconstruction_error": error,
            "threshold": detector.threshold,
            "confidence": confidence,
            "status": "FAULTY" if is_faulty else "NORMAL"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...), demo: bool = Query(False)):
    if detector is None and not (DEMO_MODE or demo):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
        try:
            if DEMO_MODE or demo:
                await file.read()
                is_faulty = random.random() > 0.7
                error = random.uniform(0.001, 0.02) if is_faulty else random.uniform(0.0001, 0.005)
                threshold = 0.005
                confidence = min(error / (threshold * 2), 1.0)
                
                results.append({
                    "filename": file.filename,
                    "is_faulty": is_faulty,
                    "reconstruction_error": error,
                    "threshold": threshold,
                    "confidence": confidence,
                    "status": "FAULTY" if is_faulty else "NORMAL",
                    "demo": True
                })
            else:
                contents = await file.read()
                
                # Load audio with scipy
                sr, waveform_np = wavfile.read(io.BytesIO(contents))
                
                # Convert to float32 normalized
                if waveform_np.dtype == np.int16:
                    waveform_np = waveform_np.astype(np.float32) / 32768.0
                elif waveform_np.dtype == np.int32:
                    waveform_np = waveform_np.astype(np.float32) / 2147483648.0
                else:
                    waveform_np = waveform_np.astype(np.float32)
                
                # Handle stereo
                if len(waveform_np.shape) > 1:
                    waveform_np = waveform_np.mean(axis=1)
                
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                
                mel_spec = preprocess_audio(waveform, sr, config).to(detector.device)
                
                with torch.no_grad():
                    reconstructed, latent = detector(mel_spec)
                    error = detector.get_reconstruction_error(mel_spec).item()
                    
                    is_faulty = error > detector.threshold
                    confidence = min(error / (detector.threshold * 2), 1.0) if detector.threshold > 0 else 0.5
                
                results.append({
                    "filename": file.filename,
                    "is_faulty": bool(is_faulty),
                    "reconstruction_error": error,
                    "threshold": detector.threshold,
                    "confidence": confidence,
                    "status": "FAULTY" if is_faulty else "NORMAL"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/threshold")
async def get_threshold():
    if detector is None and not DEMO_MODE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if DEMO_MODE:
        return {"threshold": 0.005, "demo": True}
    
    return {"threshold": detector.threshold}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config['frontend']['host'] if config else "0.0.0.0",
        port=config['frontend']['port'] if config else 8000,
        reload=True
    )