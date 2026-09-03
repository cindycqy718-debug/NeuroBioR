
import numpy as np
import scipy.signal

def predict(window):
    """
    Deterministic, zero-shot EEG source-cohort classifier for APAVA dataset.
    Input: window [256, 16] at 256 Hz, independently standardized.
    Output: [prob_healthy, prob_alzheimer]
    """
    # Ensure window is a numpy array
    x = np.asarray(window, dtype=np.float64)
    fs = 256.0
    
    # Calculate PSD using Welch's method
    # nperseg=256 gives 1Hz resolution for a 1s window
    freqs, psds = scipy.signal.welch(x, fs=fs, axis=0, nperseg=256)
    
    # Average PSD across all 16 channels
    avg_psd = np.mean(psds, axis=1)
    
    # Define frequency bands
    # Delta: 0.5-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    theta_pow = np.sum(avg_psd[theta_mask])
    alpha_pow = np.sum(avg_psd[alpha_mask])
    beta_pow = np.sum(avg_psd[beta_mask])
    
    # Total power in the range of interest
    total_pow = theta_pow + alpha_pow + beta_pow + 1e-9
    
    # Features
    # 1. Theta/Alpha Ratio (Higher in Alzheimer's)
    tar = theta_pow / (alpha_pow + 1e-9)
    
    # 2. Relative Alpha Power (Lower in Alzheimer's)
    rel_alpha = alpha_pow / total_pow
    
    # 3. Spectral Centroid (Lower in Alzheimer's due to slowing)
    mask_range = (freqs >= 4) & (freqs <= 30)
    centroid = np.sum(freqs[mask_range] * avg_psd[mask_range]) / (np.sum(avg_psd[mask_range]) + 1e-9)
    
    # Heuristic scoring based on common EEG findings in AD research
    # Healthy: Alpha dominant (~10Hz), TAR < 1.0, rel_alpha > 0.3
    # Alzheimer's: Theta increase, Alpha decrease, slowing of centroid
    
    # Normalize features into a score
    # TAR: 0.8 (H) -> 1.5 (AD)
    # Rel Alpha: 0.4 (H) -> 0.2 (AD)
    # Centroid: 12 (H) -> 9 (AD)
    
    score = 0.0
    score += (tar - 1.0) * 0.6
    score += (0.35 - rel_alpha) * 2.0
    score += (10.5 - centroid) * 0.2
    
    # Convert score to probability using sigmoid
    prob_alzheimer = 1.0 / (1.0 + np.exp(-score))
    prob_healthy = 1.0 - prob_alzheimer
    
    return np.array([float(prob_healthy), float(prob_alzheimer)])
