import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG source-cohort classifier.
    Returns [p_healthy, p_alzheimer] for a [256, 16] EEG window at 256 Hz.
    
    Based on established EEG biomarkers for Alzheimer's disease:
    - Increased slow wave activity (delta, theta)
    - Decreased alpha power
    - Spectral slowing (lower mean frequency)
    - Reduced spectral entropy
    """
    window = np.asarray(window, dtype=np.float64)
    fs = 256
    n_times, n_channels = window.shape
    
    # Compute power spectral density using Welch's method on time axis
    nperseg = min(128, n_times)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    # Compute relative band powers
    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.mean(np.sum(psd[mask], axis=0))
    
    total_power = np.sum(psd) + 1e-10
    rel_delta = band_power(0.5, 4) / total_power
    rel_theta = band_power(4, 8) / total_power
    rel_alpha = band_power(8, 13) / total_power
    rel_beta = band_power(13, 30) / total_power
    
    # Mean frequency (spectral centroid)
    mean_freq = np.sum(freqs[:, np.newaxis] * psd) / (np.sum(psd) + 1e-10)
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.mean(np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=0))
    
    # Temporal variability
    temporal_std = np.mean(np.std(window, axis=0))
    
    # Cross-channel synchronization
    if n_channels > 1:
        corr_matrix = np.corrcoef(window.T)
        mean_sync = np.mean(np.abs(corr_matrix[np.triu_indices(n_channels, k=1)]))
    else:
        mean_sync = 0.5
    
    # === Alzheimer's Disease Scoring ===
    # Based on established EEG biomarkers
    
    ad_score = 0.0
    
    # Slowing ratio: (delta + theta) / alpha
    # Higher values indicate pathological slowing (AD signature)
    slowing_ratio = (rel_delta + rel_theta) / (rel_alpha + 1e-10)
    if slowing_ratio > 1.0:
        ad_score += (slowing_ratio - 1.0) * 0.5
    
    # Alpha power deficit (normal: ~25-40%, AD: often <20%)
    if rel_alpha < 0.25:
        ad_score += (0.25 - rel_alpha) * 2.0
    
    # Delta power excess (normal: ~10-25%, AD: often >30%)
    if rel_delta > 0.25:
        ad_score += (rel_delta - 0.25) * 2.0
    
    # Mean frequency shift (normal: ~9-11 Hz, AD: often <8 Hz)
    if mean_freq < 9.0:
        ad_score += (9.0 - mean_freq) * 0.15
    
    # Spectral entropy reduction (simpler spectrum in AD)
    if spectral_entropy < 2.0:
        ad_score += (2.0 - spectral_entropy) * 0.4
    
    # Convert score to probability using logistic function
    p_alzheimer = 1.0 / (1.0 + math.exp(-(ad_score - 1.5)))
    
    # Clamp to valid probability range
    p_alzheimer = max(0.01, min(0.99, p_alzheimer))
    
    return np.array([1.0 - p_alzheimer, p_alzheimer], dtype=np.float64)
