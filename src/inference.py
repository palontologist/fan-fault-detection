import os
import yaml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import librosa

from model import load_model, CNNAutoencoder


class FanFaultDetector:
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: Optional[str] = None
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        if device is None:
            self.device = torch.device(
                self.config['training']['device'] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.model = load_model(model_path, self.config, self.device)
        self.threshold = self._load_threshold(model_path)
        
        self.sample_rate = self.config['data']['audio']['sample_rate']
        self.n_mels = self.config['data']['audio']['n_mels']
        self.n_fft = self.config['data']['audio']['n_fft']
        self.hop_length = self.config['data']['audio']['hop_length']
        self.duration = self.config['data']['audio']['duration']
    
    def _load_threshold(self, model_path: str) -> float:
        checkpoint = torch.load(model_path, map_location=self.device)
        return checkpoint.get('threshold', 0.5)
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec.unsqueeze(0)
    
    def predict(self, audio_path: str) -> Dict:
        mel_spec = self.preprocess_audio(audio_path).to(self.device)
        
        with torch.no_grad():
            reconstructed, latent = self.model(mel_spec)
            error = self.model.get_reconstruction_error(mel_spec).item()
            
            is_faulty = error > self.threshold
            confidence = min(error / (self.threshold * 2), 1.0) if self.threshold > 0 else 0.5
        
        return {
            'is_faulty': bool(is_faulty),
            'reconstruction_error': error,
            'threshold': self.threshold,
            'confidence': confidence,
            'status': 'FAULTY' if is_faulty else 'NORMAL'
        }
    
    def predict_batch(self, audio_paths: list) -> list:
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                result['file'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': path,
                    'error': str(e)
                })
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fan Fault Detection Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file or directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    detector = FanFaultDetector(args.model, args.config, args.device)
    
    audio_path = Path(args.audio)
    if audio_path.is_file():
        result = detector.predict(str(audio_path))
        print(f"File: {audio_path}")
        print(f"Status: {result['status']}")
        print(f"Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"Threshold: {result['threshold']:.6f}")
        print(f"Confidence: {result['confidence']:.4f}")
    elif audio_path.is_dir():
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(audio_path.rglob(f'*{ext}'))
        
        results = detector.predict_batch([str(f) for f in audio_files])
        
        for r in results:
            if 'error' in r:
                print(f"File: {r['file']} - Error: {r['error']}")
            else:
                print(f"File: {r['file']} - Status: {r['status']} - Error: {r['reconstruction_error']:.6f}")


if __name__ == "__main__":
    main()