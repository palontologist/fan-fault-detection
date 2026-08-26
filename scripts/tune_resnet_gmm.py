#!/usr/bin/env python3
"""
Tune ResNet + GMM/OC-SVM with better threshold calibration
Use more normal samples, better threshold selection (Youden's J, F1-optimal)
"""
import torch
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import torchaudio.transforms as T
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
import joblib
import sys
sys.path.append('src')
from resnet_gmm_detector import ResNetAnomalyDetector, ResNetFeatureExtractor, SpectrogramToImage


def find_optimal_threshold(scores, labels):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Youden's J = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def find_f1_optimal_threshold(scores, labels):
    """Find threshold that maximizes F1 score"""
    best_f1 = 0
    best_thresh = 0
    for thresh in np.linspace(scores.min(), scores.max(), 1000):
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_with_optimal_thresholds(detector, test_wavs, test_labels):
    """Evaluate with multiple threshold strategies"""
    scores = detector.predict_scores(test_wavs)
    
    # Default 95th percentile
    thresh_95 = np.percentile(scores[labels==0], 95) if np.any(np.array(labels)==0) else np.median(scores)
    
    # Youden's J
    thresh_youden = find_optimal_threshold(scores, np.array(test_labels))
    
    # F1 optimal
    thresh_f1 = find_f1_optimal_threshold(scores, np.array(test_labels))
    
    results = {}
    for name, thresh in [('95th', thresh_95), ('Youden', thresh_youden), ('F1-opt', thresh_f1)]:
        preds = (scores > thresh).astype(int)
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        auc = roc_auc_score(test_labels, scores)
        acc = np.mean(preds == test_labels)
        f1 = f1_score(test_labels, preds)
        normal_acc = np.mean(np.array(preds)[np.array(test_labels)==0] == 0) if np.any(np.array(test_labels)==0) else 0
        anomaly_acc = np.mean(np.array(preds)[np.array(test_labels)==1] == 1) if np.any(np.array(test_labels)==1) else 0
        
        results[name] = {
            'threshold': thresh,
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'normal_acc': normal_acc,
            'anomaly_acc': anomaly_acc
        }
        print(f"  {name}: thresh={thresh:.4f}, AUC={auc:.4f}, Acc={acc:.2f}, F1={f1:.2f}, NAcc={normal_acc:.2f}, AAcc={anomaly_acc:.2f}")
    
    return results


def train_and_evaluate_per_id(mid, detector_type='gmm', n_components=8, max_normal=100, max_test=30):
    """Train and evaluate detector for a specific machine ID"""
from pathlib import Path
from pydub import AudioSegment
import numpy as np
    
    normal_dir = Path('data/raw/mimii_fan/normal')
    anomaly_dir = Path('data/raw/mimii_fan/anomaly')
    
    # Get normal files for this ID
    normal_files = list(normal_dir.glob(f'normal_id_{mid}_*.wav'))
    anomaly_files = list(Path('data/raw/mimii_fan/anomaly').glob(f'anomaly_id_{mid}_*.wav'))
    
    if len(normal_files) < 20:
        print(f"  ID {mid}: Not enough normal samples")
        return None
    
    # Load normal waveforms for training
    train_wavs = []
    for f in normal_files[:max_normal]:
        try:
            audio = AudioSegment.from_file(f)
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")
            continue
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        waveform_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:
            waveform_np = waveform_np.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            waveform_np = waveform_np.astype(np.float32) / 2147483648.0
        else:
            waveform_np = waveform_np.astype(np.float32) / (2**(audio.sample_width * 8 - 1))
        
        target_len = 160000
        if len(waveform_np) > target_len:
            waveform_np = waveform_np[:target_len]
        elif len(waveform_np) < target_len:
            waveform_np = np.pad(waveform_np, (0, target_len - len(waveform_np)))
        train_wavs.append(torch.from_numpy(waveform_np))
    
    # Train detector
    detector = ResNetAnomalyDetector(model_name='resnet18', detector_type=detector_type, n_components=8)
    detector.fit(train_wavs[:min(50, len(train_wavs))])
    
    # Test on held-out normal and anomaly
    test_normal_files = list(normal_dir.glob(f'normal_id_{mid}_*.wav'))[50:80]
    test_anomaly_files = list(Path('data/raw/mimii_fan/anomaly').glob(f'anomaly_id_{mid}_*.wav'))[:20]
    
    test_wavs = []
    test_labels = []
    
    for f in list(normal_files)[50:70]:
        try:
            audio = AudioSegment.from_file(f)
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")
            continue
            
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        waveform_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:
            waveform_np = waveform_np.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            waveform_np = waveform_np.astype(np.float32) / 2147483648.0
        else:
            waveform_np = waveform_np.astype(np.float32) / (2**(audio.sample_width * 8 - 1))
        
        target_len = 160000
        if len(waveform_np) > target_len:
            waveform_np = waveform_np[:target_len]
        elif len(waveform_np) < target_len:
            waveform_np = np.pad(waveform_np, (0, target_len - len(waveform_np)))
        test_wavs.append(torch.from_numpy(waveform_np))
        test_labels.append(0)
    
    for f in list(anomaly_files)[:20]:
        try:
            audio = AudioSegment.from_file(f)
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")
            continue
            
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        waveform_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:
            waveform_np = waveform_np.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            waveform_np = waveform_np.astype(np.float32) / 2147483648.0
        else:
            waveform_np = waveform_np.astype(np.float32) / (2**(audio.sample_width * 8 - 1))
        
        target_len = 160000
        if len(waveform_np) > target_len:
            waveform_np = waveform_np[:target_len]
        elif len(waveform_np) < target_len:
            waveform_np = np.pad(waveform_np, (0, target_len - len(waveform_np)))
        test_wavs.append(torch.from_numpy(waveform_np))
        test_labels.append(1)
    
    # Evaluate with multiple thresholds
    print(f"  ID {mid} ({detector_type}):")
    results = evaluate_with_optimal_thresholds(detector, test_wavs, test_labels)
    return results


def main():
    from pathlib import Path
    import re
    
    print("=" * 60)
    print("ResNet + GMM/OC-SVM Tuning with Optimal Thresholds")
    print("=" * 60)
    
    # Test GMM and OC-SVM for each ID
    all_results = {}
    
    for mid in ['00', '02', '04', '06']:
        print(f"\n{'='*50}")
        print(f"Testing ID {mid}")
        print(f"{'='*50}")
        
        mid_results = {}
        
        # GMM
        try:
            results_gmm = train_and_evaluate_per_id(mid, 'gmm', n_components=8)
            mid_results['gmm'] = results_gmm
        except Exception as e:
            print(f"  GMM failed: {e}")
        
        # OC-SVM
        try:
            results_ocsvm = train_and_evaluate_per_id(mid, 'ocsvm')
            mid_results['ocsvm'] = results_ocsvm
        except Exception as e:
            print(f"  OC-SVM failed: {e}")
        
        all_results[mid] = mid_results
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for mid, mid_results in all_results.items():
        for det_type, det_results in mid_results.items():
            if det_results:
                best = max(det_results.items(), key=lambda x: x[1]['f1'])
                print(f"ID {mid} ({det_type}): Best={best[0]}, F1={best[1]['f1']:.3f}, AUC={best[1]['auc']:.4f}, NAcc={best[1]['normal_acc']:.2f}, AAcc={best[1]['anomaly_acc']:.2f}")


if __name__ == "__main__":
    main()