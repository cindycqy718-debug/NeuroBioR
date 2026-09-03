
import numpy as np
from scipy import signal

def predict(window):
    n_channels = window.shape[1]
    theta_ratios = []
    
    for ch in range(n_channels):
        f, Pxx = signal.periodogram(window[:, ch], fs=256)
        f = np.asarray(f)
        # Verify f is array
        if not isinstance(f, np.ndarray):
            f = np.array(f)
        theta_mask = (f >= 4) & (f < 8)
        alpha_mask = (f >= 8) & (f < 12)
        
        theta_power = np.sum(Pxx[theta_mask])
        alpha_power = np.sum(Pxx[alpha_mask])
        
        ratio = theta_power / (alpha_power + 1e-6)
        theta_ratios.append(ratio)
    
    avg_ratio = np.mean(theta_ratios)
    x0 = 0.8
    k = 10.0
    p1 = 1 / (1 + np.exp(-k * (avg_ratio - x0)))
    p0 = 1 - p1
    return [p0, p1]
