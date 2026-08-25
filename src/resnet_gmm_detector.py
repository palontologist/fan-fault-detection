#!/usr/bin/env python3
"""
ResNet Feature Extractor + GMM/OC-SVM for Fan Fault Detection
Based on: "Acoustic Anomaly Detection for Machine Sounds based on Image Transfer Learning" (2021)
Uses ResNet pretrained on ImageNet to extract features from Mel spectrograms
Anomaly detection with Gaussian Mixture Model (GMM) or One-Class SVM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio.transforms as T
import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path
from typing import Tuple, Optional, List
import joblib
import warnings
warnings.filterwarnings('ignore')


class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor for Mel spectrograms"""
    def __init__(self, model_name='resnet18', pretrained=True, feature_layer='layer4'):
        super().__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final FC layer
        self.backbone.fc = nn.Identity()
        
        # Feature layer to extract from
        self.feature_layer = feature_layer
        self._register_hook()
        
        self.features = None
    
    def _register_hook(self):
        def hook(module, input, output):
            self.features = output
        if self.feature_layer == 'layer4':
            self.backbone.layer4.register_forward_hook(hook)
        elif self.feature_layer == 'avgpool':
            self.backbone.avgpool.register_forward_hook(hook)
    
    def forward(self, x):
        """x: (batch, 3, 224, 224) - RGB images"""
        _ = self.backbone(x)
        if self.features is not None:
            # Global average pooling if needed
            if self.features.dim() == 4:
                feat = F.adaptive_avg_pool2d(self.features, (1, 1))
                return feat.view(feat.size(0), -1)
            return self.features
        return self.backbone(x)


class SpectrogramToImage(nn.Module):
    """Convert Mel spectrogram to RGB image for ResNet"""
    def __init__(self, image_size=224, colormap='viridis'):
        super().__init__()
        self.image_size = image_size
        self.colormap = colormap
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, mel_spec):
        """
        mel_spec: (batch, n_mels, n_frames) or (n_mels, n_frames)
        Returns: (batch, 3, 224, 224) RGB images
        """
        if mel_spec.dim() == 3:
            batch_size = mel_spec.shape[0]
            images = []
            for i in range(batch_size):
                img = self._spectrogram_to_rgb(mel_spec[i])
                images.append(img)
            return torch.stack(images)
        else:
            return self._spectrogram_to_rgb(mel_spec).unsqueeze(0)
    
    def _spectrogram_to_rgb(self, mel_spec):
        """Convert single mel spectrogram to RGB using colormap"""
        # mel_spec: (n_mels, n_frames)
        mel_np = mel_spec.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
        
        # Apply colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap(self.colormap)
        rgb = cmap(mel_np)[:, :, :3]  # (H, W, 3)
        
        # Resize to 224x224
        from PIL import Image
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        return self.normalize(img_tensor)


class ResNetAnomalyDetector:
    """ResNet Feature Extractor + GMM/OC-SVM for anomaly detection"""
    def __init__(self, model_name='resnet18', detector_type='gmm', n_components=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature extractor
        self.feature_extractor = ResNetFeatureExtractor(model_name, pretrained=True)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Spectrogram to image converter
        self.spec_to_image = SpectrogramToImage()
        
        # Mel spectrogram extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        )
        
        # Anomaly detector
        self.detector_type = detector_type
        if detector_type == 'gmm':
            self.detector = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        elif detector_type == 'ocsvm':
            self.detector = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        else:
            raise ValueError(f"Unknown detector: {detector_type}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None
    
    def extract_features(self, waveform):
        """Extract ResNet features from waveform"""
        # Waveform to Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Mel spectrogram to RGB image
        rgb = self.spec_to_image(mel_spec)
        
        # Extract features
        with torch.no_grad():
            rgb = rgb.to(self.device)
            features = self.feature_extractor(rgb)
        
        return features.cpu().numpy()
    
    def extract_features_batch(self, waveforms):
        """Extract features for batch of waveforms"""
        features_list = []
        for wf in waveforms:
            feat = self.extract_features(wf.unsqueeze(0))
            features_list.append(feat)
        return np.vstack(features_list)
    
    def fit(self, normal_waveforms):
        """Fit anomaly detector on normal data only"""
        print(f"Extracting features from {len(normal_waveforms)} normal samples...")
        features = self.extract_features_batch(normal_waveforms)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        print(f"Fitting {self.detector_type} detector...")
        self.detector.fit(features_scaled)
        self.is_fitted = True
        
        # Compute threshold on normal data
        if self.detector_type == 'gmm':
            scores = -self.detector.score_samples(self.scaler.transform(features))
        else:
            scores = -self.detector.decision_function(features_scaled)
        
        # 95th percentile as threshold
        self.threshold = np.percentile(scores, 95)
        print(f"Threshold set to: {self.threshold:.4f}")
        
        return self.threshold
    
    def predict(self, waveforms):
        """Predict anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        features = self.extract_features_batch(waveforms)
        features_scaled = self.scaler.transform(features)
        
        if self.detector_type == 'gmm':
            scores = -self.detector.score_samples(features_scaled)
        else:
            scores = -self.detector.decision_function(features_scaled)
        
        is_faulty = scores > self.threshold
        return scores, is_faulty
    
    def save(self, path):
        """Save detector and scaler"""
        joblib.dump({
            'detector': self.detector,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'detector_type': self.detector_type
        }, path)
    
    def load(self, path):
        """Load detector and scaler"""
        data = joblib.load(path)
        self.detector = data['detector']
        self.scaler = data['scaler']
        self.threshold = data['threshold']
        self.detector_type = data['detector_type']
        self.is_fitted = True


def prepare_waveforms_from_dir(data_dir, max_per_class=100):
    """Load waveforms from normal/anomaly directories"""
    normal_dir = Path(data_dir) / 'normal'
    anomaly_dir = Path(data_dir) / 'anomaly'
    
    normal_files = list(normal_dir.glob('*.wav'))[:max_per_class]
    anomaly_files = list(anomaly_dir.glob('*.wav'))[:max_per_class]
    
    waveforms = []
    labels = []
    
    for f in normal_files:
        sr, wav = wavfile.read(f)
        wav = wav.astype(np.float32) / 32768.0
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        # Pad/trim to 10 seconds
        target_len = 160000
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        waveforms.append(torch.from_numpy(wav))
        labels.append(0)
    
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
    
    return waveforms, np.array(labels)


def evaluate_detector(detector, test_waveforms, test_labels):
    """Evaluate detector performance"""
    scores, preds = detector.predict(test_waveforms)
    
    auc = roc_auc_score(test_labels, scores)
    accuracy = np.mean(preds == test_labels)
    
    # Per-class accuracy
    normal_mask = test_labels == 0
    anomaly_mask = test_labels == 1
    
    normal_acc = np.mean(preds[normal_mask] == 0) if normal_mask.any() else 0
    anomaly_acc = np.mean(preds[anomaly_mask] == 1) if anomaly_mask.any() else 0
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'normal_acc': normal_acc,
        'anomaly_acc': anomaly_acc,
        'scores': scores,
        'preds': preds
    }


if __name__ == "__main__":
    # Quick test
    device = torch.device('cpu')
    extractor = ResNetFeatureExtractor('resnet18')
    print(f"Feature dim: {extractor.feature_dim}")
    
    # Test feature extraction
    x = torch.randn(1, 3, 224, 224)
    feat = extractor(x)
    print(f"Feature shape: {feat.shape}")