import numpy as np
from scipy import signal

def predict(window):
    freqs, psd = signal.welch(window, fs=256, axis=0, nperseg=256)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    total_power = np.sum(psd, axis=0)
    beta_power = np.sum(psd[beta_mask], axis=0)
    beta_fraction = beta_power / total_power
    avg_beta = np.mean(beta_fraction)
    exponent = 10 * (avg_beta - 0.25)
    p1 = 1 / (1 + np.exp(-exponent))
    p0 = 1 - p1
    return np.array([p0, p1])
