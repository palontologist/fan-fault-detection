#!/usr/bin/env python3
"""
Test and compare anomaly detection approaches:
1. Current CNN Autoencoder (baseline)
2. STgram-MFN (spectral-temporal + ArcFace)
3. ResNet + GMM/OC-SVM
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from collections import defaultdict
import torchaudio.transforms as T

# Import our implementations
import sys
sys.path.append('/home/palontologist/Downloads/dev/fan-fault-detection/src')
from stgram_mfn import STgramMFN, compute_anomaly_score
from resnet_gmm_detector import ResNetAnomalyDetector, prepare_waveforms_from_dir, evaluate_detector
from model import CNNAutoencoder


def load_waveforms_from_dir(data_dir, max_per_class=20):
    """Load waveforms for testing"""
    normal_dir = Path(data_dir) / 'normal'
    anomaly_dir = Path(data_dir) / 'anomaly'
    
    normal_files = list(normal_dir.glob('*.wav'))[:max_per_class]
    anomaly_files = list(anomaly_dir.glob('*.wav'))[:max_per_class]
    
    waveforms = []
    labels = []
    filenames = []
    
    for f in normal_files:
        sr, wav = wavfile.read(f)
        wav = wav.astype(np.float32) / 32768.0
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        target_len = 160000
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        waveforms.append(torch.from_numpy(wav))
        labels.append(0)
        filenames.append(f.name)
    
    for f in anomaly_files:
        sr, wav = wavfile.read(f)
        wav = wav.astype(np.float32) / 32768.0
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        target_len = 160000
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        waveforms.append(torch.from_numpy(wav))
        labels.append(1)
        filenames.append(f.name)
    
    return waveforms, np.array(labels), filenames


def test_cnn_autoencoder(waveforms, labels, filenames):
    """Test current CNN Autoencoder"""
    print("\n=== Testing CNN Autoencoder (baseline) ===")
    
    device = torch.device('cpu')
    ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
    
    model = CNNAutoencoder(latent_dim=128, input_shape=(128, 313))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    threshold = ckpt.get('threshold', 0.5)
    
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )
    
    results = []
    for i, (wav, label) in enumerate(zip(waveforms, labels)):
        mel = mel_transform(wav.unsqueeze(0))
        mel = torch.log(mel + 1e-8)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        with torch.no_grad():
            recon, _ = model(mel)
            error = torch.mean((mel - recon) ** 2).item()
        
        pred = 1 if error > threshold else 0
        results.append({
            'filename': filenames[i],
            'true': label,
            'pred': pred,
            'error': error,
            'threshold': threshold
        })
    
    true_labels = [r['true'] for r in results]
    preds = [r['pred'] for r in results]
    
    from sklearn.metrics import roc_auc_score
    errors = [r['error'] for r in results]
    auc = roc_auc_score(true_labels, errors)
    accuracy = np.mean(np.array(preds) == np.array(true_labels))
    
    normal_mask = np.array(true_labels) == 0
    anomaly_mask = np.array(true_labels) == 1
    
    normal_acc = np.mean(np.array(preds)[normal_mask] == 0) if normal_mask.any() else 0
    anomaly_acc = np.mean(np.array(preds)[anomaly_mask] == 1) if anomaly_mask.any() else 0
    
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Normal Acc: {normal_acc:.4f}")
    print(f"  Anomaly Acc: {anomaly_acc:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'normal_acc': normal_acc,
        'anomaly_acc': anomaly_acc,
        'threshold': threshold,
        'errors': errors,
        'preds': preds
    }


def test_stgram_mfn(waveforms, labels, filenames):
    """Test STgram-MFN"""
    print("\n=== Testing STgram-MFN ===")
    
    device = torch.device('cpu')
    
    # Check if trained model exists
    model_path = 'checkpoints/stgram_mfn_best.pth'
    if not Path(model_path).exists():
        print("  No trained STgram-MFN model found. Need to train first.")
        return None
    
    model = STgramMFN(num_machine_ids=7)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Compute anomaly scores
    scores = []
    for wav in waveforms:
        score, _ = compute_anomaly_score(model, wav.unsqueeze(0), torch.device('cpu'))
        scores.append(score[0])
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    print(f"  AUC: {auc:.4f}")
    
    return {'auc': auc, 'scores': scores}


def test_resnet_gmm(waveforms, labels, filenames):
    """Test ResNet + GMM"""
    print("\n=== Testing ResNet + GMM ===")
    
    device = torch.device('cpu')
    
    # Split by ID for training
    id_waveforms = defaultdict(list)
    id_labels = defaultdict(list)
    id_filenames = defaultdict(list)
    
    for wav, label, fname in zip(waveforms, labels, filenames):
        # Extract ID from filename
        import re
        match = re.search(r'id_(\d{2})_', fname)
        if match:
            mid = match.group(1)
            id_waveforms[mid].append(wav)
            id_labels[mid].append(label)
            id_filenames[mid].append(fname)
    
    results = {}
    for mid in ['00', '02', '04', '06']:
        if mid not in id_waveforms:
            continue
        
        print(f"\n  Testing ID {mid}...")
        detector = ResNetAnomalyDetector(model_name='resnet18', detector_type='gmm', n_components=8)
        
        # Get normal waveforms for this ID
        normal_wavs = [w for w, l in zip(id_waveforms[mid], id_labels[mid]) if l == 0]
        anomaly_wavs = [w for w, l in zip(id_waveforms[mid], id_labels[mid]) if l == 1]
        
        if len(normal_wavs) < 10:
            print(f"  Not enough normal samples for ID {mid}")
            continue
        
        # Train on normal only
        detector.fit(normal_wavs[:50])  # Limit for speed
        
        # Test
        test_wavs = normal_wavs[:20] + anomaly_wavs[:20]
        test_labels = [0]*min(20, len(normal_wavs)) + [1]*min(20, len(anomaly_wavs))
        
        result = evaluate_detector(detector, test_wavs, np.array(test_labels))
        results[mid] = result
        
        print(f"  ID {mid}: AUC={result['auc']:.4f}, Normal={result['normal_acc']:.2f}, Anomaly={result['anomaly_acc']:.2f}")
    
    return results


def main():
    print("=" * 60)
    print("ANOMALY DETECTION COMPARISON")
    print("=" * 60)
    
    # Load test data for each ID
    data_dir = Path("data/raw/mimii_fan")
    
    all_results = {}
    
    for mid in ['00', '02', '04', '06']:
        print(f"\n{'='*60}")
        print(f"TESTING ID {mid}")
        print(f"{'='*60}")
        
        normal_dir = data_dir / 'normal'
        anomaly_dir = data_dir / 'anomaly'
        
        normal_files = list(normal_dir.glob(f'normal_id_{mid}_*.wav'))[:20]
        anomaly_files = list(anomaly_dir.glob(f'anomaly_id_{mid}_*.wav'))[:20]
        
        if not normal_files:
            print(f"No data for ID {mid}")
            continue
        
        waveforms = []
        labels = []
        filenames = []
        
        for f in normal_files:
            sr, wav = wavfile.read(f)
            wav = wav.astype(np.float32) / 32768.0
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            target_len = 160000
            if len(wav) > target_len:
                wav = wav[:target_len]
            elif len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            waveforms.append(torch.from_numpy(wav))
            labels.append(0)
            filenames.append(f.name)
        
        for f in anomaly_files[:20]:
            sr, wav = wavfile.read(f)
            wav = wav.astype(np.float32) / 32768.0
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            target_len = 160000
            if len(wav) > target_len:
                wav = wav[:target_len]
            elif len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            waveforms.append(torch.from_numpy(wav))
            labels.append(1)
            filenames.append(f.name)
        
        # Test all three methods
        cnn_results = test_cnn_autoencoder(waveforms, labels, filenames)
        
        # Test ResNet GMM per ID
        resnet_results = {}
        for mid in ['00', '02', '04', '06']:
            if mid == filenames[0].split('_')[2]:  # Only test current ID
                pass  # Skip for now - too slow
        
        all_results[mid] = {
            'cnn': cnn_results,
            'resnet': None,
            'stgram': None
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mid, res in all_results.items():
        if res['cnn']:
            print(f"ID {mid}: CNN AUC={res['cnn']['auc']:.4f}, Acc={res['cnn']['accuracy']:.2f}")


if __name__ == "__main__":
    main()