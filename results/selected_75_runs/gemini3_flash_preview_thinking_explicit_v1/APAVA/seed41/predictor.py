import numpy as np
from scipy import signal

def predict(window):
    # window shape: [256, 16], fs = 256 Hz
    # Axis 0 is time, Axis 1 is channels
    fs = 256
    
    # 1. Spectral Analysis using Welch
    # nperseg=128 gives 2Hz resolution, which is sufficient for band power
    f, psd = signal.welch(window, fs=fs, nperseg=128, axis=0)
    
    # 2. Extract Band Powers
    # Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    delta = np.sum(psd[(f >= 1) & (f < 4), :], axis=0)
    theta = np.sum(psd[(f >= 4) & (f < 8), :], axis=0)
    alpha = np.sum(psd[(f >= 8) & (f < 13), :], axis=0)
    beta = np.sum(psd[(f >= 13) & (f < 30), :], axis=0)
    
    # 3. Calculate Features
    # Total power in the bands of interest
    total_band_power = delta + theta + alpha + beta + 1e-12
    
    # Relative powers averaged across channels
    rel_theta = np.mean(theta / total_band_power)
    rel_alpha = np.mean(alpha / total_band_power)
    
    # Slowing Ratio: (Delta + Theta) / (Alpha + Beta)
    # AD is characterized by increased slowing
    slowing_ratio = np.mean((delta + theta) / (alpha + beta + 1e-12))
    
    # Spectral Entropy (Complexity)
    # Normalize PSD to sum to 1 across frequencies for each channel
    psd_sum = np.sum(psd, axis=0, keepdims=True) + 1e-12
    psd_norm = psd / psd_sum
    # Calculate entropy per channel and average
    ent = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
    mean_ent = np.mean(ent)
    
    # 4. Heuristic Scoring
    # These coefficients are based on general EEG findings in AD research:
    # - Higher slowing ratio (slowing_ratio > 1.0 is often a threshold)
    # - Lower alpha power (rel_alpha < 0.3)
    # - Higher theta power (rel_theta > 0.2)
    # - Lower spectral entropy (mean_ent < 3.0)
    
    # We construct a logit score for Class 1 (Alzheimer's)
    score = 0.0
    score += (slowing_ratio - 1.2) * 2.5
    score += (0.3 - rel_alpha) * 5.0
    score += (rel_theta - 0.25) * 3.0
    score += (3.2 - mean_ent) * 1.5
    
    # 5. Convert to Probabilities
    # Sigmoid function for Class 1 probability
    p1 = 1.0 / (1.0 + np.exp(-score))
    p0 = 1.0 - p1
    
    return np.array([float(p0), float(p1)], dtype=np.float64)