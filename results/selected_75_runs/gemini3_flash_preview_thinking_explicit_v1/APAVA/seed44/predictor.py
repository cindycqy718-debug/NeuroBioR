
import numpy as np
import scipy.signal
import scipy.stats
import math

def predict(window):
    """
    EEG source-cohort classifier for APAVA dataset.
    Input: window of shape [256, 16] at 256 Hz.
    Output: [prob_healthy, prob_alzheimer]
    """
    # Ensure window is a numpy array
    x = np.asarray(window, dtype=np.float64)
    fs = 256
    
    # 1. Compute Power Spectral Density using Welch's method
    # nperseg=128 gives a frequency resolution of 2 Hz
    f, psd = scipy.signal.welch(x, fs=fs, axis=0, nperseg=128)
    
    # 2. Band Power Calculation
    # Define frequency masks
    delta_mask = (f >= 0.5) & (f < 4)
    theta_mask = (f >= 4) & (f < 8)
    alpha_mask = (f >= 8) & (f < 13)
    beta_mask  = (f >= 13) & (f < 30)
    
    # Sum power in bands for each channel
    delta_p = np.sum(psd[delta_mask, :], axis=0)
    theta_p = np.sum(psd[theta_mask, :], axis=0)
    alpha_p = np.sum(psd[alpha_mask, :], axis=0)
    beta_p  = np.sum(psd[beta_mask, :], axis=0)
    
    # 3. Feature Extraction
    # Total power in the 0.5-30 Hz range
    total_p = delta_p + theta_p + alpha_p + beta_p + 1e-12
    
    # Relative powers
    rel_theta = np.mean(theta_p / total_p)
    rel_alpha = np.mean(alpha_p / total_p)
    
    # Slowing ratio: (Delta + Theta) / (Alpha + Beta)
    slowing_ratio = np.mean((delta_p + theta_p) / (alpha_p + beta_p + 1e-12))
    
    # Spectral Entropy (simplified)
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-12)
    # Avoid log(0)
    ent = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
    avg_entropy = np.mean(ent)
    
    # 4. Heuristic Scoring
    # Alzheimer's characteristics: 
    # - Higher relative theta
    # - Lower relative alpha
    # - Higher slowing ratio
    # - Lower spectral entropy
    
    # Standardized heuristic weights (based on typical EEG research findings)
    # These values are chosen to balance the features.
    score = (
        5.0 * (rel_theta - 0.25) - 
        4.0 * (rel_alpha - 0.35) + 
        1.5 * (slowing_ratio - 1.0) - 
        2.0 * (avg_entropy - 3.0)
    )
    
    # Sigmoid to get probability of Alzheimer's (class 1)
    prob_alz = 1.0 / (1.0 + math.exp(-score))
    prob_healthy = 1.0 - prob_alz
    
    return [float(prob_healthy), float(prob_alz)]
