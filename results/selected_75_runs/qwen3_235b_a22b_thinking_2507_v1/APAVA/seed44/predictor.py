import numpy as np
from scipy import signal

def predict(window):
    f, Pxx = signal.welch(window, fs=256, nperseg=128, axis=0)
    theta_idx = np.where((f >= 4) & (f <= 8))[0]
    alpha_idx = np.where((f >= 8) & (f <= 12))[0]
    theta_power = np.mean(Pxx[theta_idx, :], axis=0)
    alpha_power = np.mean(Pxx[alpha_idx, :], axis=0)
    theta_avg = np.mean(theta_power)
    alpha_avg = np.mean(alpha_power)
    ratio = (alpha_avg + 1e-5) / (theta_avg + 1e-5)
    score = 5.0 * (ratio - 1.0)
    p0 = 1.0 / (1.0 + np.exp(-score))
    p1 = 1.0 - p0
    return [p0, p1]