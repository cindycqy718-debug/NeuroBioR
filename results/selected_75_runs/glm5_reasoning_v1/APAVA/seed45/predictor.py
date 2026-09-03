import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Predict source cohort from EEG window.
    
    Parameters:
    -----------
    window : np.ndarray
        Shape [256, 16], 256 time samples at 256 Hz, 16 channels
        
    Returns:
    --------
    np.ndarray
        Shape [2], probabilities for [healthy, alzheimer]
    """
    window = np.asarray(window, dtype=np.float64)
    fs = 256
    n_samples, n_channels = window.shape
    
    # Compute PSD using Welch method (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=n_samples)
    
    # Normalize PSD to get relative power
    total_power = np.sum(psd, axis=0, keepdims=True)
    total_power = np.maximum(total_power, 1e-10)
    psd_rel = psd / total_power
    
    # Define frequency band masks
    delta_mask = (freqs >= 1) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    
    # Compute mean relative power in each band
    delta_power = float(np.mean(np.sum(psd_rel[delta_mask], axis=0)))
    theta_power = float(np.mean(np.sum(psd_rel[theta_mask], axis=0)))
    alpha_power = float(np.mean(np.sum(psd_rel[alpha_mask], axis=0)))
    
    # Alpha/delta ratio (key AD marker: lower in AD due to slowing)
    alpha_delta_ratio = alpha_power / max(delta_power, 1e-10)
    
    # Spectral entropy (complexity measure: lower in AD)
    psd_safe = np.maximum(psd_rel, 1e-10)
    entropy_per_channel = -np.sum(psd_safe * np.log2(psd_safe), axis=0)
    spectral_entropy = float(np.mean(entropy_per_channel))
    
    # Cross-channel correlation (functional connectivity: lower in AD)
    corr_matrix = np.corrcoef(window, rowvar=False)
    triu_idx = np.triu_indices(n_channels, k=1)
    correlations = corr_matrix[triu_idx]
    valid_corrs = correlations[~np.isnan(correlations)]
    if len(valid_corrs) > 0:
        mean_correlation = float(np.mean(np.abs(valid_corrs)))
    else:
        mean_correlation = 0.5
    
    # Compute AD likelihood score based on established EEG markers
    score = 0.0
    
    # Alpha/delta ratio (lower = more AD-like slowing)
    if alpha_delta_ratio < 0.3:
        score += 0.40
    elif alpha_delta_ratio < 0.6:
        score += 0.30
    elif alpha_delta_ratio < 1.0:
        score += 0.15
    elif alpha_delta_ratio < 1.5:
        score += 0.05
    
    # Theta power (higher = more AD-like)
    if theta_power > 0.35:
        score += 0.20
    elif theta_power > 0.25:
        score += 0.10
    
    # Spectral entropy (lower = less complex, more AD-like)
    if spectral_entropy < 2.0:
        score += 0.20
    elif spectral_entropy < 3.0:
        score += 0.10
    
    # Cross-channel correlation (lower = reduced connectivity, more AD-like)
    if mean_correlation < 0.2:
        score += 0.20
    elif mean_correlation < 0.4:
        score += 0.10
    
    # Convert score to probability using sigmoid
    p_ad = 1.0 / (1.0 + math.exp(-12.0 * (score - 0.35)))
    
    return np.array([1.0 - p_ad, p_ad], dtype=np.float64)
