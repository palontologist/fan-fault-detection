#!/usr/bin/env python3
"""
Improve anomaly detection by finding OPTIMAL per-ID thresholds.
Uses Youden's J statistic on held-out normal/anomaly data instead of
the broken 95th-percentile-of-normal-only approach.
"""
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import torchaudio.transforms as T
import sys
sys.path.append('src')
from model import CNNAutoencoder

mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)


def get_error(model_state_dict, wav_path, _model_cache={}):
    """Compute reconstruction error (with model caching for speed)."""
    key = id(model_state_dict)
    if key not in _model_cache:
        model = CNNAutoencoder(latent_dim=128, input_shape=(128, 313))
        model.load_state_dict(model_state_dict)
        model.eval()
        _model_cache[key] = model
    
    model = _model_cache[key]
    
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float32) / 32768.0
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    target_len = 160000
    if len(wav) > target_len:
        wav = wav[:target_len]
    elif len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    
    mel = mel_transform(wav_t)
    mel = torch.log(mel + 1e-8)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = mel.unsqueeze(0)
    
    with torch.no_grad():
        recon, _ = model(mel)
        return torch.mean((mel - recon) ** 2).item()


def find_optimal_threshold(normal_errors, anomaly_errors):
    """Find threshold maximizing balanced accuracy (Youden's J)."""
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    
    best_thresh, best_score = None, -1
    results = []
    
    for thresh in np.linspace(all_errors.min(), all_errors.max(), 200):
        preds = (all_errors > thresh).astype(int)
        tpr = preds[labels == 1].mean()   # anomaly detection rate
        tnr = 1 - preds[labels == 0].mean()  # normal detection rate
        balanced_acc = (tpr + tnr) / 2
        
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_thresh = thresh
            results = {'tpr': tpr, 'tnr': tnr, 'anomaly_acc': tpr, 'normal_acc': tnr}
    
    return best_thresh, best_score, results


def main():
    print("=" * 60)
    print("OPTIMAL THRESHOLD SEARCH PER MACHINE ID")
    print("=" * 60)
    
    optimal_thresholds = {}
    
    for mid in ['00', '02', '04', '06']:
        # Use held-out normal samples (100-150) not used in training val split,
        # plus all available anomalies
        normal_files = list(Path('data/raw/mimii_fan/normal').glob(f'normal_id_{mid}_*.wav'))[100:130]
        anomaly_files = list(Path('data/raw/mimii_fan/anomaly').glob(f'anomaly_id_{mid}_*.wav'))[:30]
        
        if len(anomaly_files) == 0:
            continue
        
        ckpt = torch.load(f'checkpoints/best_model_id_{mid}.pth', map_location='cpu')
        state_dict = ckpt['model_state_dict']
        
        print(f"\nID {mid}: computing errors ({len(normal_files)} normal, {len(anomaly_files)} anomaly)...")
        normal_errors = [get_error(state_dict, f) for f in normal_files]
        anomaly_errors = [get_error(state_dict, f) for f in anomaly_files]
        
        old_threshold = ckpt.get('threshold', 0.5)
        
        optimal_threshold, score, details = find_optimal_threshold(normal_errors, anomaly_errors)
        
        print(f"  Old threshold: {old_threshold:.4f} -> Anomaly acc: {sum(e > old_threshold for e in anomaly_errors)/len(anomaly_errors):.0%}, Normal acc: {sum(e <= old_threshold for e in normal_errors)/len(normal_errors):.0%}")
        print(f"  NEW threshold: {optimal_threshold:.4f}")
        print(f"  Balanced Acc: {score:.0%} | Anomaly det: {details['tpr']:.0%} | Normal det: {details['tnr']:.0%}")
        
        optimal_thresholds[mid] = float(optimal_threshold)
        
        # Update checkpoint with new optimal threshold
        ckpt['threshold'] = float(optimal_threshold)
        ckpt['threshold_method'] = 'youden_j_optimal'
        torch.save(ckpt, f'checkpoints/best_model_id_{mid}.pth')
        print(f"  ✓ Checkpoint updated")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Updated Thresholds")
    print("=" * 60)
    for mid, thresh in optimal_thresholds.items():
        print(f"ID {mid}: {thresh:.4f}")
    
    # Save summary for API reference
    import json
    with open('checkpoints/optimal_thresholds.json', 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print("\nSaved to checkpoints/optimal_thresholds.json")


if __name__ == "__main__":
    main()
