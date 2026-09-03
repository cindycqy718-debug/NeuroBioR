import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG source-cohort classifier.
    
    Input: window [256, 16] at 256 Hz (time x channels)
    Output: probabilities [P(healthy_source_cohort), P(alzheimer_source_cohort)]
    
    Uses established EEG markers for Alzheimer's disease:
    - Slowing of background rhythm (increased delta/theta, decreased alpha)
    - Reduced spectral entropy (more regular patterns)
    - Reduced functional connectivity (lower cross-channel correlation)
    """
    fs = 256.0
    n_samples, n_channels = window.shape
    
    # Compute PSD using Welch method (axis=0 for time axis)
    nperseg = min(64, n_samples)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    psd_avg = np.mean(psd, axis=1)
    
    # Helper for band power
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.mean(psd_avg[mask])) if np.any(mask) else 0.0
    
    # Compute band powers (Hz)
    delta = band_power(1.0, 4.0)
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 12.0)
    beta = band_power(12.0, 30.0)
    
    total = delta + theta + alpha + beta + 1e-10
    
    # Relative powers
    delta_rel = delta / total
    alpha_rel = alpha / total
    
    # Slowing index: (delta+theta)/(alpha+beta) - elevated in AD
    slowing_index = (delta + theta) / (alpha + beta + 1e-10)
    
    # Spectral entropy - reduced in AD (more regular/slow patterns)
    psd_norm = psd_avg / (np.sum(psd_avg) + 1e-10)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    
    # Cross-channel correlation - reduced in AD (reduced connectivity)
    if n_channels > 1:
        corr_matrix = np.corrcoef(window.T)
        upper_tri_idx = np.triu_indices(n_channels, k=1)
        mean_correlation = float(np.mean(np.abs(corr_matrix[upper_tri_idx])))
    else:
        mean_correlation = 0.5
    
    # Temporal variability (coefficient of variation across time)
    temporal_cv = float(np.mean(np.std(window, axis=0) / (np.mean(np.abs(window), axis=0) + 1e-10)))
    
    # Normalize features to [0,1] based on typical EEG ranges
    # Higher values indicate more AD-like characteristics
    
    # Slowing index: typical 0.3-2.5, higher = more AD
    s_slow = min(1.0, max(0.0, (slowing_index - 0.4) / 2.0))
    
    # Alpha relative power: typical 0.15-0.45, lower = more AD
    s_alpha = min(1.0, max(0.0, (0.40 - alpha_rel) / 0.25))
    
    # Spectral entropy: typical 2.5-5.0, lower = more AD
    s_entropy = min(1.0, max(0.0, (4.5 - spectral_entropy) / 2.0))
    
    # Mean correlation: typical 0.15-0.65, lower = more AD
    s_corr = min(1.0, max(0.0, (0.55 - mean_correlation) / 0.40))
    
    # Weighted composite AD score
    # Weights based on literature: slowing and alpha are strongest markers
    ad_score = (
        0.35 * s_slow +    # Slowing index - most reliable
        0.30 * s_alpha +   # Alpha power - well established
        0.15 * s_entropy + # Complexity measure
        0.20 * s_corr      # Connectivity measure
    )
    
    # Convert to probability using logistic function
    # Centered at 0.5 with moderate slope
    logit = 5.0 * (ad_score - 0.5)
    prob_ad = 1.0 / (1.0 + math.exp(-logit))
    prob_healthy = 1.0 - prob_ad
    
    return np.array([prob_healthy, prob_ad], dtype=np.float64)
