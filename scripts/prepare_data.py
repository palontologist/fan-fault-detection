#!/usr/bin/env python3
"""
Data preparation script - organizes downloaded datasets into train/val/test splits
"""
import os
import shutil
import random
from pathlib import Path
import yaml

def organize_dataset(raw_dir: str, processed_dir: str, splits: dict = None):
    if splits is None:
        splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    normal_files = list((raw_path / 'normal').glob('*.wav')) + list((raw_path / 'normal').glob('*.mp3')) + list((raw_path / 'normal').glob('*.flac'))
    anomaly_files = list((raw_path / 'anomaly').glob('*.wav')) + list((raw_path / 'anomaly').glob('*.mp3')) + list((raw_path / 'anomaly').glob('*.flac'))
    
    print(f"Found {len(normal_files)} normal files, {len(anomaly_files)} anomaly files")
    
    random.seed(42)
    random.shuffle(normal_files)
    random.shuffle(anomaly_files)
    
    def split_files(files, splits):
        n = len(files)
        train_end = int(n * splits['train'])
        val_end = train_end + int(n * splits['val'])
        
        return {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }
    
    normal_splits = split_files(normal_files, splits)
    anomaly_splits = split_files(anomaly_files, splits)
    
    for split_name in ['train', 'val', 'test']:
        for label_name, label_files in [('normal', normal_splits), ('anomaly', anomaly_splits)]:
            split_dir = processed_path / split_name / label_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for f in label_files[split_name]:
                dst = split_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
            
            print(f"  {split_name}/{label_name}: {len(label_files[split_name])} files")

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['data'].get('raw_dir', 'data/raw')
    processed_dir = config['data'].get('data_dir', 'data/processed')
    
    datasets = ['kaggle_fan', 'mimii_fan', 'dcase2022_fan']
    
    for dataset in datasets:
        dataset_raw = Path(raw_dir) / dataset
        dataset_processed = Path(processed_dir) / dataset
        
        if dataset_raw.exists():
            print(f"\nOrganizing {dataset}...")
            organize_dataset(str(dataset_raw), str(dataset_processed))
        else:
            print(f"\nDataset {dataset} not found at {dataset_raw}")
            print(f"Expected structure:")
            print(f"  {dataset_raw}/")
            print(f"    ├── normal/")
            print(f"    └── anomaly/")

if __name__ == "__main__":
    main()