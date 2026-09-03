
import numpy as np
from scipy.signal import periodogram

def predict(window):
    n_time, n_channels = window.shape
    if n_time != 256 or n_channels != 33:
        raise ValueError("Invalid input shape")
    
    beta_ratios = []
    for j in range(n_channels):
        f, Pxx = periodogram(window[:, j], fs=256, axis=0)
        beta_mask = (f >= 13) & (f <= 30)
        beta_power = np.sum(Pxx[beta_mask])
        total_power = np.sum(Pxx)
        beta_ratio = beta_power / total_power if total_power > 0 else 0.0
        beta_ratios.append(beta_ratio)
    
    beta_avg = np.mean(beta_ratios)
    threshold = 0.15
    slope = 10.0
    score = slope * (beta_avg - threshold)
    p1 = 1.0 / (1.0 + np.exp(-score))
    return [1.0 - p1, p1]
