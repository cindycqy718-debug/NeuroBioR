import numpy as np
import scipy.signal
import math

def predict(window):
    # window shape: [256, 33], fs: 256 Hz
    # Axis 0 is time, Axis 1 is channels
    fs = 256
    
    # 1. Compute Power Spectral Density using Welch's method
    # nperseg=256 gives 1 Hz frequency resolution
    f, psd = scipy.signal.welch(window, fs=fs, axis=0, nperseg=256)
    
    # 2. Define frequency bands
    # Parkinson's is often associated with "slowing" of EEG: 
    # increased power in Delta (1-4Hz) and Theta (4-8Hz)
    # and decreased power in Alpha (8-13Hz) and Beta (13-30Hz).
    
    mask_low = (f >= 1.0) & (f <= 8.0)
    mask_high = (f > 8.0) & (f <= 30.0)
    
    # Average power across channels and then across frequency bins
    # psd shape is [freq_bins, channels]
    low_power = np.mean(psd[mask_low, :])
    high_power = np.mean(psd[mask_high, :])
    
    # 3. Calculate Ratio (Low/High)
    # A higher ratio suggests Parkinson's (more slowing)
    ratio = low_power / (high_power + 1e-9)
    
    # 4. Logistic mapping to probabilities
    # We use a heuristic threshold. In standardized EEG, 
    # a ratio significantly above 1.0 often indicates slowing.
    # Center the sigmoid around 1.1
    threshold = 1.1
    steepness = 3.0
    
    logit = (ratio - threshold) * steepness
    
    # Sigmoid function for probability of class 1 (Parkinson's)
    p1 = 1.0 / (1.0 + math.exp(-logit))
    p0 = 1.0 - p1
    
    return [float(p0), float(p1)]