#!/usr/bin/env python3
"""
Per-Machine-ID Training Script for Fan Fault Detection.
Trains separate models for each MIMII fan ID (00, 02, 04, 06).
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import torchaudio.transforms as T
import re
from collections import defaultdict


class FanAudioDataset(Dataset):
    def __init__(self, file_list, config):
        self.config = config
        self.audio_cfg = config['data']['audio']
        self.sample_rate = self.audio_cfg['sample_rate']
        self.duration = self.audio_cfg['duration']
        self.n_mels = self.audio_cfg['n_mels']
        self.n_fft = self.audio_cfg['n_fft']
        self.hop_length = self.audio_cfg['hop_length']
        
        self.target_length = int(self.sample_rate * self.duration)
        
        self.files = file_list  # List of (filepath, label)
        
        # Mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        print(f"  Dataset: {len([f for f,l in self.files if l==0])} normal, {len([f for f,l in self.files if l==1])} anomaly")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        
        # Load with scipy
        sr, waveform_np = wavfile.read(filepath)
        
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
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad/trim
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Normalize per sample
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec, torch.tensor(label, dtype=torch.long)


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=1, input_shape=(128, 313)):
        super().__init__()
        self.input_shape = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.final_upsample = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h)
    
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 4, 4)
        h = self.decoder(h)
        return self.final_upsample(h)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=[1, 2, 3])
        return error


def extract_machine_id(filename):
    """Extract machine ID from MIMII filename."""
    # Pattern: normal_id_00_00000000.wav or anomaly_id_04_00000000.wav
    match = re.search(r'id_(\d{2})_', filename)
    if match:
        return match.group(1)
    return None


def get_files_by_id(data_dir):
    """Organize files by machine ID."""
    data_dir = Path(data_dir)
    
    normal_files = list((data_dir / "normal").glob("*.wav"))
    anomaly_files = list((data_dir / "anomaly").glob("*.wav"))
    
    by_id = defaultdict(lambda: {"normal": [], "anomaly": []})
    
    for f in normal_files:
        mid = extract_machine_id(f.name)
        if mid:
            by_id[mid]["normal"].append((f, 0))
    
    for f in anomaly_files:
        mid = extract_machine_id(f.name)
        if mid:
            by_id[mid]["anomaly"].append((f, 1))
    
    return by_id


def train_model_for_id(machine_id, normal_files, anomaly_files, config, epochs=30, batch_size=16, lr=1e-3, device="cpu"):
    """Train a model for a specific machine ID."""
    print(f"\n{'='*60}")
    print(f"Training model for Fan ID: {machine_id}")
    print(f"  Normal samples: {len(normal_files)}, Anomaly samples: {len(anomaly_files)}")
    print(f"{'='*60}")
    
    # Combine all files for this ID
    all_files = normal_files + anomaly_files
    np.random.shuffle(all_files)
    
    # Split train/val (80/20)
    n_train = int(0.8 * len(all_files))
    n_val = len(all_files) - n_train
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    
    # Create datasets
    train_dataset = FanAudioDataset(train_files, config)
    val_dataset = FanAudioDataset(val_files, config)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = CNNAutoencoder(latent_dim=config['training']['model']['latent_dim']).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # Checkpoint dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'machine_id': machine_id,
            }, ckpt_dir / f"best_model_id_{machine_id}.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
    
    # Compute threshold on normal validation samples
    print(f"\nComputing anomaly threshold for ID {machine_id}...")
    model.eval()
    normal_errors = []
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to(device)
            errors = model.get_reconstruction_error(batch).cpu().numpy()
            for err, label in zip(errors, labels):
                if label == 0:  # normal
                    normal_errors.append(err)
    
    if normal_errors:
        threshold = float(np.percentile(normal_errors, config['training']['loss']['anomaly_threshold_percentile']))
    else:
        threshold = 0.5
    
    # Save final model with threshold
    checkpoint = torch.load(ckpt_dir / f"best_model_id_{machine_id}.pth", map_location=device)
    checkpoint['threshold'] = threshold
    checkpoint['normal_errors_stats'] = {
        'mean': float(np.mean(normal_errors)) if normal_errors else 0,
        'std': float(np.std(normal_errors)) if normal_errors else 0,
        'min': float(np.min(normal_errors)) if normal_errors else 0,
        'max': float(np.max(normal_errors)) if normal_errors else 0,
    }
    torch.save(checkpoint, ckpt_dir / f"best_model_id_{machine_id}.pth")
    
    print(f"ID {machine_id} Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Threshold (95th percentile): {threshold:.6f}")
    print(f"  Model saved to: {ckpt_dir / f'best_model_id_{machine_id}.pth'}")
    
    return model, threshold, normal_errors


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cpu"
    data_dir = "data/raw/mimii_fan"
    
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Loading and organizing data by machine ID...")
    by_id = get_files_by_id(data_dir)
    
    if not by_id:
        print("No files found with valid machine IDs!")
        return
    
    print(f"Found machine IDs: {sorted(by_id.keys())}")
    for mid in sorted(by_id.keys()):
        print(f"  ID {mid}: {len(by_id[mid]['normal'])} normal, {len(by_id[mid]['anomaly'])} anomaly")
    
    # Train model for each ID
    results = {}
    for machine_id in sorted(by_id.keys()):
        normal_files = by_id[machine_id]["normal"]
        anomaly_files = by_id[machine_id]["anomaly"]
        
        if len(normal_files) < 10:
            print(f"Skipping ID {machine_id}: insufficient normal samples ({len(normal_files)})")
            continue
        
        model, threshold, errors = train_model_for_id(
            machine_id, normal_files, anomaly_files, config,
            epochs=30, batch_size=16, lr=1e-3, device=device
        )
        results[machine_id] = {
            'threshold': threshold,
            'num_normal': len(normal_files),
            'num_anomaly': len(anomaly_files),
            'error_stats': {'mean': np.mean(errors), 'std': np.std(errors)}
        }
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for mid, r in results.items():
        print(f"ID {mid}: threshold={r['threshold']:.4f}, normal={r['num_normal']}, anomaly={r['num_anomaly']}")
    
    print("\nAll per-ID models trained and saved!")


if __name__ == "__main__":
    main()