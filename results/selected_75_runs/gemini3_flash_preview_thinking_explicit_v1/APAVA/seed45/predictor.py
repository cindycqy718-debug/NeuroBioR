import numpy as np
from scipy import signal
import math

def predict(window):
    # window shape: [256, 16], fs = 256 Hz
    # Axis 0 is time, Axis 1 is channels
    fs = 256
    
    # Calculate Power Spectral Density
    # nperseg=256 gives a frequency resolution of 1 Hz
    f, pxx = signal.welch(window, fs=fs, axis=0, nperseg=256)
    
    # Define bands
    # Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    delta = np.sum(pxx[(f >= 1) & (f < 4), :], axis=0)
    theta = np.sum(pxx[(f >= 4) & (f < 8), :], axis=0)
    alpha = np.sum(pxx[(f >= 8) & (f < 13), :], axis=0)
    beta = np.sum(pxx[(f >= 13) & (f < 30), :], axis=0)
    
    # Alzheimer's is characterized by "slowing": increase in Delta/Theta, decrease in Alpha/Beta
    # We use the ratio of (Delta + Theta) to (Alpha + Beta)
    # Avoid division by zero
    numerator = delta + theta
    denominator = alpha + beta + 1e-9
    ratio = numerator / denominator
    
    # Average ratio across all 16 channels
    mean_ratio = np.mean(ratio)
    
    # Log-transform the ratio to make it more symmetric
    # A ratio of 1.0 (equal power) maps to 0.0
    # Healthy subjects typically have mean_ratio < 1.0 (strong Alpha)
    # AD subjects typically have mean_ratio > 1.0 (stronger slow waves)
    log_ratio = math.log10(mean_ratio + 1e-9)
    
    # Threshold and scaling for sigmoid
    # These values are chosen based on general EEG-AD literature
    # Healthy: log_ratio < 0, AD: log_ratio > 0
    threshold = -0.1  # Slightly offset to favor healthy if ratio is exactly 1
    scale = 5.0      # Sensitivity of the classifier
    
    p1 = 1.0 / (1.0 + math.exp(-(log_ratio - threshold) * scale))
    p0 = 1.0 - p1
    
    return np.array([float(p0), float(p1)], dtype=np.float64)