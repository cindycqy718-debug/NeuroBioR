import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Zero-shot EEG source-cohort classifier for TDBRAIN dataset.
    
    Classifies standardized EEG windows as healthy (class 0) or 
    Parkinson's (class 1) source cohort based on spectral biomarkers.
    
    Parameters
    ----------
    window : np.ndarray
        Shape [256, 33] array containing 1 second of 33-channel EEG
        at 256 Hz, independently standardized.
    
    Returns
    -------
    np.ndarray
        Shape [2] array with probabilities for [healthy, parkinson]
        that sum to 1.0.
    """
    fs = 256.0
    window = np.asarray(window, dtype=np.float64)
    n_samples, n_channels = window.shape
    
    # Compute power spectral density using Welch method
    freqs, psd = signal.welch(window, fs=fs, nperseg=128, axis=0)
    
    # Convert to relative power (normalized per channel)
    psd_sum = np.sum(psd, axis=0, keepdims=True) + 1e-10
    psd_rel = psd / psd_sum
    
    # Helper function for band power
    def band_power(f, p, f_low, f_high):
        mask = (f >= f_low) & (f <= f_high)
        if not np.any(mask):
            return np.zeros(p.shape[1])
        return np.mean(p[mask], axis=0)
    
    # Compute band powers per channel, then average across channels
    theta_ch = band_power(freqs, psd_rel, 4.0, 8.0)
    alpha_ch = band_power(freqs, psd_rel, 8.0, 13.0)
    beta_ch = band_power(freqs, psd_rel, 13.0, 30.0)
    delta_ch = band_power(freqs, psd_rel, 1.0, 4.0)
    
    theta = np.mean(theta_ch)
    alpha = np.mean(alpha_ch)
    beta = np.mean(beta_ch)
    delta = np.mean(delta_ch)
    
    # Feature 1: Theta/Alpha ratio (elevated in Parkinson's)
    ta_ratio = theta / (alpha + 1e-10)
    
    # Feature 2: Alpha power fraction (reduced in Parkinson's)
    alpha_frac = alpha
    
    # Feature 3: Beta/Alpha ratio
    ba_ratio = beta / (alpha + 1e-10)
    
    # Feature 4: Spectral entropy (complexity measure)
    psd_safe = psd_rel + 1e-10
    entropy_ch = -np.sum(psd_safe * np.log(psd_safe), axis=0)
    entropy = np.mean(entropy_ch) / np.log(len(freqs))
    
    # Feature 5: Inter-channel correlation with safe handling
    # Compute correlation using manual calculation to handle edge cases
    window_centered = window - np.mean(window, axis=0, keepdims=True)
    std_vals = np.std(window, axis=0) + 1e-10
    window_norm = window_centered / std_vals
    corr_matrix = np.dot(window_norm.T, window_norm) / n_samples
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_channels, k=1)
    corr_vals = np.abs(corr_matrix[triu_idx])
    corr_mean = float(np.mean(corr_vals)) if len(corr_vals) > 0 else 0.5
    
    # Feature 6: Delta/Theta ratio (slow wave activity)
    dt_ratio = delta / (theta + 1e-10)
    
    # Compute Parkinson's likelihood score based on literature-established effects
    # TA ratio: healthy ~0.4-0.9, PD ~0.9-1.8 (higher = PD-like)
    ta_score = np.clip((ta_ratio - 0.4) / 1.4, 0.0, 1.0)
    
    # Alpha fraction: healthy ~0.15-0.30, PD ~0.08-0.18 (lower = PD-like)
    alpha_score = np.clip((0.25 - alpha_frac) / 0.17, 0.0, 1.0)
    
    # BA ratio: healthy ~0.3-0.8, PD ~0.6-1.2 (higher = PD-like)
    ba_score = np.clip((ba_ratio - 0.3) / 0.9, 0.0, 1.0)
    
    # Entropy: healthy ~0.7-0.9, PD ~0.5-0.75 (lower = PD-like)
    entropy_score = np.clip((0.85 - entropy) / 0.35, 0.0, 1.0)
    
    # Correlation: moderate connectivity changes in PD
    corr_score = np.clip((corr_mean - 0.2) / 0.6, 0.0, 1.0)
    
    # Delta/Theta ratio: elevated slow activity in PD
    dt_score = np.clip((dt_ratio - 0.5) / 1.5, 0.0, 1.0)
    
    # Weighted combination based on literature evidence strength
    weights = np.array([0.28, 0.22, 0.12, 0.18, 0.10, 0.10])
    scores = np.array([ta_score, alpha_score, ba_score, entropy_score, corr_score, dt_score])
    
    composite = float(np.dot(weights, scores))
    
    # Map to probability using logistic function
    logit = 4.0 * (composite - 0.5)
    prob_parkinson = 1.0 / (1.0 + math.exp(-logit))
    
    # Ensure valid output
    prob_healthy = 1.0 - prob_parkinson
    
    return np.array([prob_healthy, prob_parkinson], dtype=np.float64)
