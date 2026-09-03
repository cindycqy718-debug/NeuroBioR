import numpy as np
from scipy import signal

def predict(window):
    fs = 256
    f, Pxx = signal.welch(window, fs=fs, nperseg=256, axis=0, window='hann')
    
    # Clinically established bands (APA guidelines)
    theta_band = (f >= 4) & (f < 8)   # Theta (4-8 Hz) - elevated in Alzheimer's
    alpha_band = (f >= 8) & (f <= 13)  # Alpha (8-13 Hz) - reduced in Alzheimer's
    
    theta_power = np.sum(Pxx[theta_band, :])
    alpha_power = np.sum(Pxx[alpha_band, :])
    
    # Numerical stability for zero-power cases
    total = theta_power + alpha_power
    if total < 1e-10:
        return [0.5, 0.5]
    
    # Avoid division by zero with epsilon (clinically negligible)
    ratio = theta_power / (alpha_power + 1e-10)
    
    # Sigmoidal mapping to probability space (centered at ratio=1.0)
    p1 = 1 / (1 + np.exp(-5 * (ratio - 1.0)))
    return [1 - p1, p1]
