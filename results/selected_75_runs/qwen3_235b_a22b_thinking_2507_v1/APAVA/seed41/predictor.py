
import numpy as np
from scipy import signal

def predict(window):
    fs = 256
    nperseg = 128
    theta_total = 0.0
    alpha_total = 0.0
    
    for ch in range(window.shape[1]):
        f, Pxx = signal.welch(window[:, ch], fs=fs, nperseg=nperseg, axis=0)
        theta_mask = (f >= 4) & (f < 8)
        alpha_mask = (f >= 8) & (f <= 12)
        theta_power = np.sum(Pxx[theta_mask])
        alpha_power = np.sum(Pxx[alpha_mask])
        theta_total += theta_power
        alpha_total += alpha_power
    
    total = theta_total + alpha_total
    epsilon = 1e-10
    r = theta_total / (total + epsilon)
    return np.array([1.0 - r, r], dtype=np.float64)
