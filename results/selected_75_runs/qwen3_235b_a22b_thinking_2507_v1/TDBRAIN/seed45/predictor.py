import numpy as np
from scipy import signal

def predict(window):
    f, Pxx = signal.welch(window, fs=256, axis=0, nperseg=256)
    alpha_mask = (f >= 8) & (f <= 12)
    beta_mask = (f >= 13) & (f <= 30)
    alpha_power = np.sum(Pxx[alpha_mask, :], axis=0)
    beta_power = np.sum(Pxx[beta_mask, :], axis=0)
    ratio = beta_power / (alpha_power + beta_power + 1e-9)
    avg_ratio = np.mean(ratio)
    return np.array([1.0 - avg_ratio, avg_ratio])
