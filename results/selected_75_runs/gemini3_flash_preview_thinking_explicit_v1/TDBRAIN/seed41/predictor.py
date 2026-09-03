
import numpy as np
from scipy import signal

def predict(window):
    """
    EEG source-cohort classifier for TDBRAIN dataset.
    Class 0: healthy_source_cohort
    Class 1: parkinson_source_cohort
    """
    # window shape: [256, 33], fs: 256 Hz
    fs = 256
    
    # Ensure window is a numpy array
    window = np.asarray(window)
    
    # Compute Power Spectral Density (PSD) using Welch's method
    # nperseg=128 gives 2Hz resolution. 
    # axis=0 is the time axis.
    f, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # Average PSD across all 33 channels
    avg_psd = np.mean(psd, axis=1)
    
    # Define frequency bands
    # Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    theta_mask = (f >= 4) & (f < 8)
    alpha_mask = (f >= 8) & (f < 13)
    beta_mask = (f >= 13) & (f < 30)
    
    # Calculate band powers
    theta_pow = np.sum(avg_psd[theta_mask])
    alpha_pow = np.sum(avg_psd[alpha_mask])
    beta_pow = np.sum(avg_psd[beta_mask])
    
    # PD heuristic: Increased Theta power, decreased Alpha and Beta power.
    # We use the ratio of Theta to (Alpha + Beta)
    # In healthy subjects, Alpha is usually dominant.
    # In PD, the spectrum shifts towards lower frequencies (slowing).
    
    numerator = theta_pow + 1e-9
    denominator = alpha_pow + beta_pow + 1e-9
    ratio = numerator / denominator
    
    # Heuristic thresholding
    # A ratio > 0.4-0.5 is often indicative of slowing in PD.
    # We use a logistic function to map the ratio to a probability.
    # Center the sigmoid at 0.45.
    logit = 5.0 * (ratio - 0.45)
    p1 = 1.0 / (1.0 + np.exp(-logit))
    
    # Ensure p1 is finite and in [0, 1]
    p1 = np.clip(p1, 0.0, 1.0)
    p0 = 1.0 - p1
    
    return np.array([p0, p1], dtype=float)
