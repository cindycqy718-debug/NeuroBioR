import numpy as np
from scipy import signal

def predict(window):
    freqs, psd = signal.welch(window, fs=256, axis=0, nperseg=256)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    beta_power = np.sum(psd[beta_mask], axis=0)
    total_power = np.sum(psd, axis=0)
    beta_prop = beta_power / total_power
    mean_beta_prop = np.mean(beta_prop)
    threshold = 0.22
    prob_class_1 = 0.65 if mean_beta_prop > threshold else 0.35
    return np.array([1.0 - prob_class_1, prob_class_1], dtype=np.float64)
