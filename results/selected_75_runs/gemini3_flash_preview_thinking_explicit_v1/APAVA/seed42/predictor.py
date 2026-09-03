
import numpy as np
from scipy import signal

def predict(window):
    """
    EEG source-cohort classifier for APAVA dataset.
    Input: window [256, 16] standardized EEG at 256 Hz.
    Output: [p_healthy, p_alzheimer]
    """
    fs = 256.0
    
    # Compute Power Spectral Density (PSD) using Welch's method
    # nperseg=128 provides 2 Hz resolution, better for short 256-sample windows
    f, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # Define frequency bands (Hz)
    # AD is characterized by 'slowing': increase in Delta/Theta, decrease in Alpha/Beta
    delta_mask = (f >= 0.5) & (f < 4.0)
    theta_mask = (f >= 4.0) & (f < 8.0)
    alpha_mask = (f >= 8.0) & (f < 13.0)
    beta_mask = (f >= 13.0) & (f < 30.0)
    
    # Calculate average power in each band across all channels
    def get_band_power(mask):
        if not np.any(mask):
            return 1e-7
        return np.mean(psd[mask, :], axis=0)

    delta_pow = get_band_power(delta_mask)
    theta_pow = get_band_power(theta_mask)
    alpha_pow = get_band_power(alpha_mask)
    beta_pow = get_band_power(beta_mask)
    
    # Feature: Ratio of slow activity (Delta + Theta) to fast activity (Alpha + Beta)
    # This ratio is typically elevated in Alzheimer's patients.
    slow_fast_ratio = (delta_pow + theta_pow) / (alpha_pow + beta_pow + 1e-7)
    
    # Average ratio across all 16 channels
    avg_ratio = np.mean(slow_fast_ratio)
    
    # Logistic mapping: Healthy (class 0) vs Alzheimer (class 1)
    # Threshold 1.2 is chosen based on typical EEG power distributions
    threshold = 1.2
    steepness = 1.5
    logit = steepness * (avg_ratio - threshold)
    
    # Sigmoid to get probability of class 1 (Alzheimer)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    p0 = 1.0 - p1
    
    return np.array([p0, p1], dtype=np.float64)
