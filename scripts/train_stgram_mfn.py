#!/usr/bin/env python3
"""
Train STgram-MFN on MIMII fan dataset
Self-supervised: predict machine ID from normal sounds
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
import re
from scipy.io import wavfile
from pydub import AudioSegment
import torchaudio.transforms as T

# Import our model
import sys
sys.path.append('src')
from stgram_mfn import STgramMFN


class STgramDataset(Dataset):
    def __init__(self, data_dir, config, machine_ids=None):
        self.config = config
        self.audio_cfg = config['data']['audio']
        self.sample_rate = self.audio_cfg['sample_rate']
        self.duration = self.audio_cfg['duration']
        self.n_mels = self.audio_cfg['n_mels']
        self.n_fft = self.audio_cfg['n_fft']
        self.hop_length = self.audio_cfg['hop_length']
        
        self.target_length = int(self.sample_rate * self.duration)
        
        # Get all files
        normal_dir = Path(data_dir) / "normal"
        anomaly_dir = Path(data_dir) / "anomaly"
        
        normal_files = list(normal_dir.glob("*.wav"))
        anomaly_files = list(anomaly_dir.glob("*.wav"))
        
        all_files = [(f, 0) for f in normal_files] + [(f, 1) for f in anomaly_files]
        
        # Filter by machine IDs if specified
        if machine_ids:
            all_files = [(f, l) for f, l in all_files if self._extract_id(f) in machine_ids]
        
        # For STgram-MFN, we only use normal data for training
        # Use all data but label with machine ID
        self.files = all_files
        
        # Mel transform for Sgram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        print(f"Loaded {len(self.files)} files")
    
    def _extract_id(self, filepath):
        match = re.search(r'id_(\d{2})_', filepath.name)
        return match.group(1) if match else None
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        
        # Load with pydub (supports MP3, WAV, FLAC, OGG, etc.)
        try:
            audio = AudioSegment.from_file(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {filepath}: {e}")
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate
        audio = audio.set_frame_rate(self.sample_rate)
        
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
        if len(waveform_np) > self.target_length:
            waveform_np = waveform_np[:self.target_length]
        elif len(waveform_np) < self.target_length:
            waveform_np = np.pad(waveform_np, (0, self.target_length - len(waveform_np)))
        
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        # Extract machine ID
        machine_id = self._extract_id(filepath)
        if machine_id:
            machine_id = int(machine_id)
        else:
            machine_id = 0
        
        return waveform, torch.tensor(machine_id, dtype=torch.long)


def train_stgram_mfn(config, data_dir, epochs=50, batch_size=16, lr=1e-4, device="cpu", machine_ids=None):
    # Dataset
    dataset = STgramDataset(data_dir, config, machine_ids=machine_ids)
    
    # Split train/val (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = STgramMFN(num_machine_ids=7).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Checkpoint dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Training STgram-MFN on {device} for {epochs} epochs...")
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            _, logits = model(batch, labels)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = batch.to(device)
                labels = labels.to(device)
                
                _, logits = model(batch, labels)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
            }, ckpt_dir / "stgram_mfn_best.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {ckpt_dir / 'stgram_mfn_best.pth'}")
    
    return model


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cpu"
    data_dir = "data/raw/mimii_fan"
    
    if not Path(data_dir).exists():
        print("No data found!")
        return
    
    # Train on IDs with anomaly data: 00, 02, 04, 06
    train_stgram_mfn(config, data_dir, epochs=50, batch_size=16, lr=1e-4, device=device, machine_ids=['00', '02', '04', '06'])


if __name__ == "__main__":
    main()