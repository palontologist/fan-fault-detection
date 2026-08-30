#!/usr/bin/env python3
"""
Generate synthetic fan audio for quick training/testing.
Normal: steady tonal components + broadband noise
Faulty: added impulses, frequency modulation, bearing noise
"""
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile


SAMPLE_RATE = 16000
DURATION = 10.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_NORMAL = 200
N_FAULTY = 50


def generate_normal_fan():
    """Generate normal fan sound: steady tones + broadband noise"""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    
    # Fundamental frequency (fan blade pass frequency ~ 50-200 Hz)
    f0 = np.random.uniform(80, 120)
    
    # Harmonics with decreasing amplitude
    signal = np.zeros(N_SAMPLES)
    for h in range(1, 8):
        amp = 1.0 / (h ** 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        signal += amp * np.sin(2 * np.pi * h * f0 * t + phase)
    
    # Broadband noise (turbulence)
    noise = np.random.normal(0, 0.1, N_SAMPLES)
    
    # Low-frequency modulation (slow speed variation)
    mod = 1 + 0.02 * np.sin(2 * np.pi * 0.1 * t)
    
    # High-frequency whine (bearing/motor)
    whine = 0.05 * np.sin(2 * np.pi * np.random.uniform(2000, 4000) * t)
    
    signal = (signal + noise + whine) * mod
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7
    return signal.astype(np.float32)


def generate_faulty_fan():
    """Generate faulty fan sound: normal + fault signatures"""
    signal = generate_normal_fan()
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    
    fault_type = np.random.choice(['impulse', 'fm', 'bearing', 'loose', 'combined'])
    
    if fault_type in ['impulse', 'combined']:
        # Periodic impulses (blade damage, contamination)
        impulse_rate = np.random.uniform(5, 20)  # Hz
        impulse_times = np.arange(0, DURATION, 1/impulse_rate)
        for it in impulse_times:
            idx = int(it * SAMPLE_RATE)
            end_idx = min(idx + 200, N_SAMPLES)
            decay_len = end_idx - idx
            if decay_len > 10:
                decay = np.exp(-np.arange(decay_len) * 50)
                impulse_shape = np.random.normal(0, 1, decay_len) * decay
                signal[idx:end_idx] += impulse_shape * np.random.uniform(0.3, 0.8)
    
    if fault_type in ['fm', 'combined']:
        # Frequency modulation (unbalance, misalignment)
        fm_depth = np.random.uniform(2, 8)
        fm_rate = np.random.uniform(0.5, 3)
        fm = fm_depth * np.sin(2 * np.pi * fm_rate * t)
        # Apply to fundamental
        f0 = 100  # approximate
        for h in range(1, 4):
            phase_mod = fm * h
            # Re-synthesize with FM (simplified)
            signal += 0.1 * np.sin(2 * np.pi * h * f0 * t + phase_mod)
    
    if fault_type in ['bearing', 'combined']:
        # Bearing fault: characteristic frequencies with harmonics
        bpfi = np.random.uniform(80, 150)  # Ball pass frequency inner
        for h in range(1, 6):
            amp = 0.05 / h
            signal += amp * np.sin(2 * np.pi * h * bpfi * t + np.random.uniform(0, 2*np.pi))
    
    if fault_type in ['loose', 'combined']:
        # Loose parts: subharmonics, rattling
        signal += 0.1 * np.sin(2 * np.pi * 25 * t)  # subharmonic
        # Random rattles
        for _ in range(np.random.randint(3, 10)):
            idx = np.random.randint(0, N_SAMPLES - 500)
            end_idx = min(idx + 500, N_SAMPLES)
            rattle_len = end_idx - idx
            signal[idx:end_idx] += np.random.normal(0, 0.2, rattle_len) * np.exp(-np.arange(rattle_len) * 10)
    
    # Re-normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7
    return signal.astype(np.float32)


def save_wav(signal, path):
    """Save signal as WAV file"""
    # Convert to int16
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, signal_int16)


def main():
    out_dir = Path("data/processed/synthetic")
    normal_dir = out_dir / "normal"
    faulty_dir = out_dir / "faulty"
    normal_dir.mkdir(parents=True, exist_ok=True)
    faulty_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {N_NORMAL} normal samples...")
    for i in tqdm(range(N_NORMAL)):
        signal = generate_normal_fan()
        save_wav(signal, normal_dir / f"normal_{i:04d}.wav")
    
    print(f"Generating {N_FAULTY} faulty samples...")
    for i in tqdm(range(N_FAULTY)):
        signal = generate_faulty_fan()
        save_wav(signal, faulty_dir / f"faulty_{i:04d}.wav")
    
    print(f"Done! Data saved to {out_dir}")
    print(f"  Normal: {len(list(normal_dir.glob('*.wav')))} files")
    print(f"  Faulty: {len(list(faulty_dir.glob('*.wav')))} files")


if __name__ == "__main__":
    main()