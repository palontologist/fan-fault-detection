#!/usr/bin/env python3
"""
STgram-MFN Implementation for Fan Fault Detection
Based on: "Anomalous Sound Detection Using Spectral-Temporal Information Fusion" (ICASSP 2022)
Architecture: TgramNet (temporal) + Sgram (spectral) -> STgram -> MobileFaceNet + ArcFace
Self-supervised: Predict machine ID from normal sounds only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Tuple, Optional
import math


class TgramNet(nn.Module):
    """Temporal feature extraction network (TgramNet) from raw waveform"""
    def __init__(self, n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Large kernel 1D conv matching Mel spectrogram parameters
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_mels, kernel_size=n_fft, stride=hop_length, padding=n_fft//2, bias=False),
            nn.BatchNorm1d(n_mels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Three CNN blocks (no dimension change)
        self.blocks = nn.Sequential(
            self._make_block(n_mels, n_mels, kernel_size=3),
            self._make_block(n_mels, n_mels, kernel_size=3),
            self._make_block(n_mels, n_mels, kernel_size=3),
        )
    
    def _make_block(self, in_ch, out_ch, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        """
        x: (batch, 1, waveform_length)
        Returns: Tgram (batch, n_mels, n_frames)
        """
        tgram = self.conv1(x)
        tgram = self.blocks(tgram)
        return tgram


class SgramExtractor(nn.Module):
    """Log-Mel Spectrogram extractor (Sgram)"""
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
    
    def forward(self, x):
        """
        x: (batch, 1, waveform_length)
        Returns: Log-Mel spectrogram (batch, n_mels, n_frames)
        """
        sgram = self.mel_transform(x)
        sgram = torch.log(sgram + 1e-8)
        # Normalize per sample
        sgram = (sgram - sgram.mean(dim=(-2, -1), keepdim=True)) / (sgram.std(dim=(-2, -1), keepdim=True) + 1e-8)
        return sgram


class STgramFusion(nn.Module):
    """Spectral-Temporal Fusion (STgram) - concatenates Sgram and Tgram"""
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.sgram_extractor = SgramExtractor(sample_rate, n_mels, n_fft, hop_length)
        self.tgram_net = TgramNet(n_mels, n_fft, hop_length)
    
    def forward(self, x):
        """
        x: (batch, 1, waveform_length)
        Returns: STgram (batch, 2, n_mels, n_frames)
        """
        sgram = self.sgram_extractor(x)  # (B, 1, M, N)
        tgram = self.tgram_net(x)        # (B, M, N)
        tgram = tgram.unsqueeze(1)       # (B, 1, M, N)
        
        # Ensure same temporal dimension
        if sgram.shape[-1] != tgram.shape[-1]:
            # Interpolate tgram to match sgram
            tgram = F.interpolate(tgram, size=sgram.shape[-1], mode='linear', align_corners=False)
        
        # Stack: (B, 2, M, N)
        stgram = torch.cat([sgram, tgram], dim=1)
        return stgram


class MobileFaceNet(nn.Module):
    """MobileFaceNet adapted for STgram input (2 channels, 128x313)"""
    def __init__(self, num_classes=7, embedding_size=512):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        
        # Depthwise separable blocks
        self.layer1 = self._make_layer(64, 64, 5, stride=2)   # 64x313 -> 32x157
        self.layer2 = self._make_layer(64, 128, 5, stride=2)  # 32x157 -> 16x79
        self.layer3 = self._make_layer(128, 128, 5, stride=2) # 16x79 -> 8x40
        self.layer4 = self._make_layer(128, 256, 5, stride=2) # 8x40 -> 4x20
        
        # Global depthwise conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),  # Pointwise
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final embedding
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # ArcFace classifier
        self.arcface = ArcFace(embedding_size, num_classes, s=30.0, m=0.5)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(self._block(in_ch if i == 0 else out_ch, out_ch, s))
        return nn.Sequential(*layers)
    
    def _block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False, groups=out_ch),  # Depthwise
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False),  # Pointwise
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x, labels=None):
        """
        x: (batch, 2, 128, 313) - STgram
        labels: (batch,) - machine IDs for ArcFace
        Returns: embeddings, logits (if labels provided)
        """
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        
        # L2 normalize embeddings
        embeddings = F.normalize(x, p=2, dim=1)
        
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return embeddings, logits
        return embeddings


class ArcFace(nn.Module):
    """ArcFace loss for better intra-class compactness and inter-class separation"""
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, embeddings, labels):
        """
        embeddings: (batch, embedding_size) - L2 normalized
        labels: (batch,) - class indices
        Returns: scaled logits (batch, num_classes)
        """
        # cos(theta)
        cosine = F.linear(embeddings, F.normalize(self.weight))
        
        # cos(theta + m)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        return output


class STgramMFN(nn.Module):
    """Complete STgram-MFN model for training"""
    def __init__(self, num_machine_ids=7, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.stgram = STgramFusion(sample_rate, n_mels, n_fft, hop_length)
        self.mfn = MobileFaceNet(num_classes=num_machine_ids)
    
    def forward(self, waveform, labels=None):
        """
        waveform: (batch, 1, waveform_length)
        labels: (batch,) machine IDs
        Returns: embeddings, logits (if labels provided)
        """
        stgram = self.stgram(waveform)  # (B, 2, 128, 313)
        return self.mfn(stgram, labels)
    
    def get_embeddings(self, waveform):
        """Get embeddings for inference (no labels needed)"""
        stgram = self.stgram(waveform)
        return self.mfn(stgram)


# Utility: Training loss with ArcFace
def stgram_mfn_loss(logits, labels):
    """Cross-entropy loss with ArcFace logits"""
    return F.cross_entropy(logits, labels)


# Anomaly scoring for inference
def compute_anomaly_score(model, waveform, device, threshold=None):
    """
    Compute anomaly score using negative log probability of predicted class
    """
    model.eval()
    with torch.no_grad():
        waveform = waveform.to(device)
        # Get embeddings and logits
        stgram = model.stgram(waveform)
        embeddings = model.mfn(stgram)
        
        # Get logits using cosine similarity with class weights (no labels needed)
        # Use the ArcFace weight matrix directly for cosine similarity
        weight_norm = F.normalize(model.mfn.arcface.weight)
        embeddings_norm = F.normalize(embeddings)
        cosine = F.linear(embeddings_norm, weight_norm)
        
        probs = F.softmax(cosine, dim=1)
        
        # Anomaly score = -log(max_prob) = -log(max_class_probability)
        max_probs, _ = probs.max(dim=1)
        anomaly_scores = -torch.log(max_probs + 1e-8)
        
        if threshold is not None:
            is_faulty = anomaly_scores > threshold
        else:
            is_faulty = None
        
        return anomaly_scores.cpu().numpy(), is_faulty


if __name__ == "__main__":
    # Quick test
    device = torch.device("cpu")
    model = STgramMFN(num_machine_ids=7)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(2, 1, 160000)  # 10 seconds at 16kHz
    embeddings, logits = model(x, labels=torch.tensor([0, 1]))
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test inference
    scores, _ = compute_anomaly_score(model, x, device)
    print(f"Anomaly scores: {scores}")