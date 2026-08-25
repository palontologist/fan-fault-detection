#!/usr/bin/env python3
"""
Download MIMII fan dataset from Zenodo.
Total size: ~1.1 GB
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil


# MIMII dataset URLs (from zenodo.org/records/3384388)
# These are the direct download links for fan machine
URLS = {
    "dev_data_fan.zip": "https://zenodo.org/records/3384388/files/dev_data_fan.zip",
}

# Alternative: DCASE 2023 Task 2 fan (smaller subset)
DCASE_URLS = {
    "dev_data_fan.zip": "https://zenodo.org/records/6355122/files/dev_data_fan.zip",
}


def download_file(url, dest_path):
    """Download with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path, extract_dir):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def main():
    data_dir = Path("data/raw/mimii_fan")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("MIMII Fan Dataset Download")
    print("=" * 50)
    print("Source: Zenodo (https://zenodo.org/records/3384388)")
    print("Size: ~1.1 GB")
    print("Structure: dev_data_fan/")
    print("  - train/normal/ (1000 normal files)")
    print("  - test/normal/ (50 normal files)")
    print("  - test/anomaly/ (50 anomalous files)")
    print()
    
    # Check if already downloaded
    zip_path = data_dir / "dev_data_fan.zip"
    if zip_path.exists():
        print(f"Found existing: {zip_path}")
        resp = input("Re-download? (y/N): ")
        if resp.lower() != 'y':
            print("Skipping download.")
        else:
            zip_path.unlink()
    
    if not zip_path.exists():
        print("Downloading...")
        try:
            download_file(URLS["dev_data_fan.zip"], zip_path)
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nTrying DCASE 2023 mirror...")
            try:
                download_file(DCASE_URLS["dev_data_fan.zip"], zip_path)
                print("Download complete from DCASE mirror!")
            except Exception as e2:
                print(f"DCASE download also failed: {e2}")
                return
    
    # Extract
    extract_dir = data_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_zip(zip_path, extract_dir)
    
    # Organize into expected structure
    print("Organizing files...")
    raw_normal = data_dir / "normal"
    raw_anomaly = data_dir / "anomaly"
    raw_normal.mkdir(exist_ok=True)
    raw_anomaly.mkdir(exist_ok=True)
    
    # Find and copy files
    for wav_file in extract_dir.rglob("*.wav"):
        rel = wav_file.relative_to(extract_dir)
        if "anomaly" in str(rel) or "abnormal" in str(rel):
            dest = raw_anomaly / wav_file.name
        else:
            dest = raw_normal / wav_file.name
        shutil.copy2(wav_file, dest)
    
    print(f"Done! Organized:")
    print(f"  Normal: {len(list(raw_normal.glob('*.wav')))} files")
    print(f"  Anomaly: {len(list(raw_anomaly.glob('*.wav')))} files")
    print(f"\nData ready at: {data_dir}")
    print("\nTo train on this data, modify quick_train.py to use:")
    print(f"  data_dir = Path('data/raw/mimii_fan')")


if __name__ == "__main__":
    main()