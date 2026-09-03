import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG classifier for AD/FTD/Healthy.
    Input: [256, 19] array at 256 Hz
    Output: [3] probabilities [healthy_control, FTD, AD]
    
    Based on established EEG biomarkers:
    - Healthy: alpha-dominant (8-13 Hz), low slowing, high complexity/connectivity
    - AD: marked slowing (delta/theta increase), alpha reduction, reduced entropy/complexity
    - FTD: intermediate pattern between healthy and AD
    """
    fs = 256
    window = np.asarray(window, dtype=np.float64)
    n_chan = window.shape[1]
    
    # Clip for numerical stability
    window = np.clip(window, -1e6, 1e6)
    
    # Spectral features via Welch (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    def band_power(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        return np.mean(psd[mask], axis=0) if np.any(mask) else np.zeros(n_chan)
    
    delta = np.mean(band_power(0.5, 4))
    theta = np.mean(band_power(4, 8))
    alpha = np.mean(band_power(8, 13))
    beta = np.mean(band_power(13, 30))
    
    # Feature 1: Slowing index (delta+theta)/(alpha+beta) - elevated in AD
    slow_idx = (delta + theta) / (alpha + beta + 1e-10)
    
    # Feature 2: Alpha fraction - reduced in AD
    alpha_frac = alpha / (delta + theta + alpha + beta + 1e-10)
    
    # Feature 3: Spectral entropy - reduced in AD (more regular slow patterns)
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    entropy = np.mean(-np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=0))
    
    # Feature 4: Hjorth complexity - reduced in AD
    d1 = np.diff(window, axis=0)
    d2 = np.diff(d1, axis=0)
    v0 = np.var(window, axis=0) + 1e-10
    v1 = np.var(d1, axis=0) + 1e-10
    v2 = np.var(d2, axis=0) + 1e-10
    complexity = np.mean(np.sqrt(v2 / v1) / np.sqrt(v1 / v0))
    
    # Feature 5: Cross-channel correlation - reduced in AD
    R = np.corrcoef(window.T)
    R = np.nan_to_num(R, nan=0.0)
    mean_corr = np.mean(np.abs(R[~np.eye(n_chan, dtype=bool)]))
    
    # Sigmoid helper for threshold-based scoring
    def sig(x, c, s):
        z = max(-20.0, min(20.0, (x - c) / s))
        return 1.0 / (1.0 + math.exp(-z))
    
    # Gaussian helper for intermediate values (FTD pattern)
    def gauss(x, c, s):
        return math.exp(-min(20.0, ((x - c) / s) ** 2))
    
    # Healthy score: low slowing, high alpha, high entropy/complexity/correlation
    H = (sig(2.0 - slow_idx, 0, 0.5) +
         sig(alpha_frac, 0.35, 0.08) +
         sig(entropy, 4.0, 0.5) +
         sig(complexity, 1.2, 0.25) +
         sig(mean_corr, 0.4, 0.1))
    
    # AD score: high slowing, low alpha, reduced entropy/complexity/correlation
    A = (sig(slow_idx, 2.5, 0.5) +
         sig(0.22 - alpha_frac, 0, 0.06) +
         sig(3.2 - entropy, 0, 0.5) +
         sig(0.85 - complexity, 0, 0.2) +
         sig(0.28 - mean_corr, 0, 0.08))
    
    # FTD score: intermediate pattern (Gaussian centered between healthy and AD)
    F = (gauss(slow_idx, 2.0, 0.6) +
         gauss(alpha_frac, 0.28, 0.08) +
         gauss(entropy, 3.5, 0.5) +
         gauss(complexity, 1.0, 0.2) +
         gauss(mean_corr, 0.33, 0.08))
    
    # Normalize to probabilities
    total = H + F + A + 1e-10
    probs = np.array([H, F, A]) / total
    
    return probs
