
import numpy as np
from scipy import signal

def predict(window):
    freqs, psd = signal.welch(window, fs=256, nperseg=256, axis=0)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    total_mask = (freqs >= 1) & (freqs <= 128)
    
    beta_power = np.sum(psd[beta_mask], axis=0)
    total_power = np.sum(psd[total_mask], axis=0)
    
    ratio = np.divide(beta_power, total_power, out=np.zeros_like(beta_power), where=total_power != 0)
    avg_ratio = np.mean(ratio)
    
    score = 10 * (avg_ratio - 0.25)
    p1 = 1 / (1 + np.exp(-score))
    p0 = 1 - p1
    return np.array([p0, p1], dtype=np.float64)
