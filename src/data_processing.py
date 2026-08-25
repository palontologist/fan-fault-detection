import os
import yaml
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm


class FanSoundDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        config: dict,
        split: str = "train",
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        
        self.sample_rate = config['data']['audio']['sample_rate']
        self.n_mels = config['data']['audio']['n_mels']
        self.n_fft = config['data']['audio']['n_fft']
        self.hop_length = config['data']['audio']['hop_length']
        self.duration = config['data']['audio']['duration']
        self.n_frames = config['data']['audio']['n_frames']
        
        self.normalize = config['data']['preprocessing']['normalize']
        self.augment_config = config['data']['preprocessing']['augment']
        
        self.files = self._collect_files()
        self.labels = self._get_labels()
    
    def _collect_files(self) -> List[Path]:
        files = []
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir
        
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            files.extend(split_dir.rglob(f'*{ext}'))
        
        return sorted(files)
    
    def _get_labels(self) -> List[int]:
        labels = []
        for f in self.files:
            if 'anomaly' in f.name.lower() or 'abnormal' in f.name.lower() or 'fault' in f.name.lower():
                labels.append(1)
            else:
                labels.append(0)
        return labels
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.files[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(str(file_path))
        
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
        
        mel_spec = self._compute_mel_spectrogram(waveform)
        
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        if self.split == "train" and self.augment_config.get('enabled', False):
            mel_spec = self._augment(mel_spec)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec.unsqueeze(0), label
    
    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        return mel_spec
    
    def _augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if np.random.random() < 0.5:
            noise = torch.randn_like(mel_spec) * self.augment_config.get('noise_factor', 0.005)
            mel_spec = mel_spec + noise
        
        if np.random.random() < 0.3:
            stretch_factor = 1 + np.random.uniform(
                -self.augment_config.get('time_stretch', 0.2),
                self.augment_config.get('time_stretch', 0.2)
            )
            mel_spec = torch.nn.functional.interpolate(
                mel_spec.unsqueeze(0),
                scale_factor=stretch_factor,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            if mel_spec.shape[1] > self.n_frames:
                mel_spec = mel_spec[:, :self.n_frames]
            elif mel_spec.shape[1] < self.n_frames:
                padding = self.n_frames - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        
        return mel_spec


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = FanSoundDataset(
        config['data']['data_dir'],
        config,
        split="train"
    )
    val_dataset = FanSoundDataset(
        config['data']['data_dir'],
        config,
        split="val"
    )
    test_dataset = FanSoundDataset(
        config['data']['data_dir'],
        config,
        split="test"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def download_kaggle_dataset(dataset_name: str, download_dir: str):
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_dir,
            unzip=True
        )
        print(f"Downloaded {dataset_name} to {download_dir}")
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Please download manually from Kaggle and place in data/raw/")


def download_zenodo_dataset(record_id: str, download_dir: str):
    import requests
    import zipfile
    import io
    
    url = f"https://zenodo.org/api/records/{record_id}/files-archive"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(download_dir)
        print(f"Downloaded Zenodo record {record_id} to {download_dir}")
    else:
        print(f"Error downloading from Zenodo: {response.status_code}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['data']['data_dir'], exist_ok=True)
    
    for dataset in config['data']['datasets']:
        if dataset['type'] == 'kaggle':
            download_kaggle_dataset(dataset['url'].split('/')[-1], config['data']['data_dir'])
        elif dataset['type'] == 'zenodo':
            record_id = dataset['url'].split('/')[-1]
            download_zenodo_dataset(record_id, config['data']['data_dir'])