#!/usr/bin/env python3
"""
Quick CPU training script for fan fault detection.
Trains CNN autoencoder on synthetic data in ~5-10 minutes.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import torchaudio
import torchaudio.transforms as T


class FanAudioDataset(Dataset):
    def __init__(self, data_dir, config, split="train"):
        self.config = config
        self.audio_cfg = config['data']['audio']
        self.sample_rate = self.audio_cfg['sample_rate']
        self.duration = self.audio_cfg['duration']
        self.n_mels = self.audio_cfg['n_mels']
        self.n_fft = self.audio_cfg['n_fft']
        self.hop_length = self.audio_cfg['hop_length']
        
        self.target_length = int(self.sample_rate * self.duration)
        
        # Get all files - for anomaly detection, train ONLY on normal
        normal_files = list((data_dir / "normal").glob("*.wav"))
        faulty_files = list((data_dir / "faulty").glob("*.wav"))
        
        # Use only normal files for training
        self.files = [(f, 0) for f in normal_files]
        np.random.shuffle(self.files)
        
        print(f"Loaded {len(normal_files)} normal, {len(faulty_files)} faulty (test only) files")
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        print(f"Loaded {len(normal_files)} normal, {len(faulty_files)} faulty files")
    
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
        
        # mel_spec is [1, n_mels, time], return as is (batch dim added by DataLoader)
        return mel_spec, torch.tensor(label, dtype=torch.long)


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=1, input_shape=(128, 313)):
        super().__init__()
        self.input_shape = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64x157
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x79
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x40
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x20
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )
        
        # Final upsample to match input
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


def train_model(config, data_dir, epochs=20, batch_size=16, lr=1e-3, device="cpu"):
    # Dataset
    dataset = FanAudioDataset(data_dir, config)
    
    # Split train/val (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
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
    train_losses = []
    val_losses = []
    
    print(f"Training on {device} for {epochs} epochs...")
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
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
            }, ckpt_dir / "best_model.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
    
    # Compute threshold on normal validation samples
    print("\nComputing anomaly threshold...")
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
        threshold = np.percentile(normal_errors, config['training']['loss']['anomaly_threshold_percentile'])
    else:
        threshold = 0.5
    
    # Save final model with threshold
    checkpoint = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    checkpoint['threshold'] = float(threshold)
    torch.save(checkpoint, ckpt_dir / "best_model.pth")
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Threshold (95th percentile): {threshold:.6f}")
    print(f"Model saved to: {ckpt_dir / 'best_model.pth'}")
    
    return model, threshold


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cpu"
    data_dir = Path("data/raw/mimii_fan")
    
    if not data_dir.exists():
        print("No data found. Run generate_synthetic_data.py first!")
        return
    
    train_model(config, data_dir, epochs=20, batch_size=16, lr=1e-3, device=device)


if __name__ == "__main__":
    main()