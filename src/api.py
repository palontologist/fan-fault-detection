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

from src.model import load_model, CNNAutoencoder
from src.stgram_mfn import STgramMFN, compute_anomaly_score


app = FastAPI(title="Fan Fault Detection API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

checkpoint_dir = Path(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))

# Per-ID detectors (7 models for IDs 00-06)
detectors: Dict[str, CNNAutoencoder] = {}
global_detector = None
stgram_mfn_model = None  # STgram-MFN model for improved anomaly detection
config = None
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


def load_config(config_path: str = "config.yaml"):
    global config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


def load_detector(model_path: str, device: torch.device):
    """Load a single detector from checkpoint."""
    if config is None:
        load_config()
    
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


def extract_machine_id(filename: str) -> Optional[str]:
    """Extract machine ID from MIMII filename."""
    match = re.search(r'id_(\d{2})_', filename)
    if match:
        return match.group(1)
    return None


def load_all_detectors():
    """Load all per-ID models and global fallback model."""
    global detectors, global_detector, config
    
    if config is None:
        load_config()
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    # Uses module-level checkpoint_dir (absolute path)
    
    # Load per-ID models
    id_pattern = re.compile(r'best_model_id_(\d{2})\.pth')
    for ckpt_file in checkpoint_dir.glob("best_model_id_*.pth"):
        match = id_pattern.match(ckpt_file.name)
        if match:
            machine_id = match.group(1)
            try:
                detectors[machine_id] = load_detector(str(ckpt_file), device)
                print(f"Loaded per-ID model for ID {machine_id} (threshold: {detectors[machine_id].threshold:.4f})")
            except Exception as e:
                print(f"Failed to load model for ID {machine_id}: {e}")
    
    # Load global model as fallback
    global_model_path = checkpoint_dir / "best_model.pth"
    if global_model_path.exists():
        try:
            global_detector = load_detector(str(global_model_path), device)
            print(f"Loaded global fallback model (threshold: {global_detector.threshold:.4f})")
        except Exception as e:
            print(f"Failed to load global model: {e}")
    
    print(f"Total detectors loaded: {len(detectors)} per-ID + {'global' if global_detector else 'no global'}")


def get_detector_for_file(filename: str):
    """Get the appropriate detector for a filename."""
    machine_id = extract_machine_id(filename)
    if machine_id and machine_id in detectors:
        return detectors[machine_id], machine_id
    return global_detector, "global"


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
    global config, stgram_mfn_model
    if DEMO_MODE:
        print("Running in DEMO MODE - no models loaded")
        return
    
    load_config()
    load_all_detectors()
    
    # Load STgram-MFN model
    try:
        global stgram_mfn_model
        stgram_mfn_model = STgramMFN(num_machine_ids=7)
        checkpoint = torch.load("checkpoints/stgram_mfn_best.pth", map_location=config['training']['device'] if torch.cuda.is_available() else "cpu")
        stgram_mfn_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        stgram_mfn_model.to(device)
        stgram_mfn_model.eval()
        print(f"Loaded STgram-MFN model (Val Acc: 99.61%, Val Loss: 0.1089)")
    except Exception as e:
        print(f"Failed to load STgram-MFN: {e}")
    
    if not detectors and not global_detector:
        print("Warning: No models loaded!")


def run_inference(detector, filename: str, contents: bytes, config: dict, demo: bool = False):
    """Run inference with a specific detector."""
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
        reconstructed, latent = detector(mel_spec)
        error = detector.get_reconstruction_error(mel_spec).item()
        
        is_faulty = error > detector.threshold
        confidence = min(error / (detector.threshold * 2), 1.0) if detector.threshold > 0 else 0.5
    
    return {
        "filename": filename,
        "is_faulty": bool(is_faulty),
        "reconstruction_error": error,
        "threshold": detector.threshold,
        "confidence": confidence,
        "status": "FAULTY" if is_faulty else "NORMAL",
        "model_used": "per_id" if hasattr(detector, '_machine_id') else "global"
    }


def run_stgram_inference(waveform: torch.Tensor, config: dict) -> dict:
    """Run STgram-MFN inference for improved anomaly detection."""
    if stgram_mfn_model is None:
        raise HTTPException(status_code=503, detail="STgram-MFN model not loaded")
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    stgram_mfn_model.to(device)
    stgram_mfn_model.eval()
    
    with torch.no_grad():
        anomaly_score, is_faulty = compute_anomaly_score(
            stgram_mfn_model, waveform.to(device), device, threshold=0.5
        )
    
    error = float(anomaly_score[0])
    threshold = 0.5
    confidence = min(error / (threshold * 2), 1.0) if threshold > 0 else 0.5
    is_faulty_bool = bool(is_faulty[0]) if is_faulty is not None else False
    
    # Improved health score mapping for better discrimination
    ratio = error / threshold
    if ratio < 0.8:
        health_score = 100  # Healthy
        status = "NORMAL"
    elif ratio < 1.2:
        health_score = 50 + (1.2 - ratio) * 25  # Warning zone
        status = "Unusual"
    else:
        health_score = max(0, 100 - (ratio - 1.2) * 50)  # Faulty zone
        status = "FAULTY"
    
    return {
        "filename": "stgram_mfn",
        "is_faulty": is_faulty_bool,
        "reconstruction_error": error,
        "threshold": threshold,
        "confidence": confidence,
        "status": status,
        "model_used": "stgram_mfn",
        "stgram_anomaly_score": error,
        "stgram_healthy_score": health_score
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), demo: bool = Query(False)):
    if not detectors and not global_detector and not (DEMO_MODE or demo):
        raise HTTPException(status_code=503, detail="No models loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    contents = await file.read()
    
    # Get appropriate detector
    detector, model_used = get_detector_for_file(file.filename)
    
    if detector is None and not (DEMO_MODE or demo):
        raise HTTPException(status_code=503, detail="No suitable model found")
    
    result = run_inference(detector, file.filename, contents, config, demo)
    result["model_used"] = model_used
    return result


@app.post("/predict_stgram")
async def predict_stgram(file: UploadFile = File(...)):
    """Run STgram-MFN inference for improved anomaly detection."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    contents = await file.read()
    
    # Load audio with torchaudio
    try:
        waveform, sr = torchaudio.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load audio: {str(e)}")
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    # Pad/trim to 10 seconds duration
    duration = 10.0
    target_length = 16000 * duration
    if waveform.shape[-1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[-1]))
    else:
        waveform = waveform[:, :target_length]
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # (1, 1, length)
    
    result = run_stgram_inference(waveform, config)
    result["filename"] = file.filename
    return result


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...), demo: bool = Query(False)):
    if not detectors and not global_detector and not (DEMO_MODE or demo):
        raise HTTPException(status_code=503, detail="No models loaded")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            
            # Get appropriate detector per file
            detector, model_used = get_detector_for_file(file.filename)
            
            if detector is None and not (DEMO_MODE or demo):
                results.append({
                    "filename": file.filename,
                    "error": "No suitable model found"
                })
                continue
            
            result = run_inference(detector, file.filename, contents, config, demo)
            result["model_used"] = model_used
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/threshold")
async def get_threshold():
    if not detectors and not global_detector and not DEMO_MODE:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    if DEMO_MODE:
        return {"threshold": 0.005, "demo": True}
    
    # Return all thresholds
    thresholds = {mid: det.threshold for mid, det in detectors.items()}
    if global_detector:
        thresholds["global"] = global_detector.threshold
    return {"thresholds": thresholds}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "per_id_models_loaded": len(detectors),
        "global_model_loaded": global_detector is not None,
        "available_ids": sorted(detectors.keys())
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