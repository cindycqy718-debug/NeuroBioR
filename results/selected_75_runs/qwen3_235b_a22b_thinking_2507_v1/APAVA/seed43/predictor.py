import numpy as np
from scipy import signal

def predict(window):
    f, Pxx = signal.periodogram(window, fs=256, axis=0)
    theta_idx = np.where((f >= 4) & (f <= 7))[0]
    alpha_idx = np.where((f >= 8) & (f <= 12))[0]
    theta_power = np.sum(Pxx[theta_idx], axis=0)
    alpha_power = np.sum(Pxx[alpha_idx], axis=0)
    epsilon = 1e-9
    ratios = theta_power / (theta_power + alpha_power + epsilon)
    avg_ratio = np.mean(ratios)
    return np.array([1.0 - avg_ratio, avg_ratio], dtype=np.float64)
