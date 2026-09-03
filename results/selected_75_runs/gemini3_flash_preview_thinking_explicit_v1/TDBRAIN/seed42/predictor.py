import numpy as np
from scipy import signal

def predict(window):
    """
    EEG source-cohort classifier for TDBRAIN dataset.
    Class 0: healthy_source_cohort
    Class 1: parkinson_source_cohort
    
    Input: window of shape [256, 33] at 256 Hz (1 second).
    Output: [prob_0, prob_1]
    """
    fs = 256
    # window shape is [time, channels] = [256, 33]
    
    # 1. Compute Power Spectral Density (PSD) using Welch's method
    # Using nperseg=256 to cover the full 1-second window for 1Hz resolution.
    f, psd = signal.welch(window, fs=fs, axis=0, nperseg=256)
    
    # 2. Average PSD across all 33 channels
    # psd shape: [129, 33] -> [129]
    psd_avg = np.mean(psd, axis=1)
    
    # 3. Extract power in specific bands
    # Parkinson's disease is often characterized by 'slowing' of EEG:
    # increased power in Delta (1-4Hz) and Theta (4-8Hz) bands,
    # and decreased power in Alpha (8-13Hz) and Beta (13-30Hz) bands.
    
    mask_slow = (f >= 1.0) & (f < 8.0)
    mask_fast = (f >= 8.0) & (f <= 30.0)
    
    # Use mean power in these ranges to handle potential bin differences
    slow_power = np.mean(psd_avg[mask_slow])
    fast_power = np.mean(psd_avg[mask_fast])
    
    # 4. Calculate the Slow-to-Fast Ratio
    # PD patients typically show a higher ratio than healthy controls.
    ratio = slow_power / (fast_power + 1e-9)
    
    # 5. Map the ratio to a probability using a logistic function
    # Based on literature, a ratio > 1.0 is more indicative of slowing.
    # We'll set the decision threshold (midpoint) at 1.1.
    # The steepness k=3.0 provides a smooth transition.
    midpoint = 1.1
    k = 3.0
    
    # Logistic function: 1 / (1 + exp(-k * (x - x0)))
    prob_1 = 1.0 / (1.0 + np.exp(-k * (ratio - midpoint)))
    prob_0 = 1.0 - prob_1
    
    return np.array([float(prob_0), float(prob_1)], dtype=np.float64)