
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def bandpower(data, sf, band, axis=0):
        band = np.asarray(band)
        freqs, psd = welch(data, sf, axis=axis)
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.sum(psd[idx_band], axis=0)

    def compute_entropy(data, axis=0):
        return entropy(np.abs(data), axis=axis)

    # Sampling frequency
    sf = 256

    # Frequency bands
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    # Compute band powers
    delta_power = np.mean(bandpower(window, sf, delta_band, axis=0))
    theta_power = np.mean(bandpower(window, sf, theta_band, axis=0))
    alpha_power = np.mean(bandpower(window, sf, alpha_band, axis=0))
    beta_power = np.mean(bandpower(window, sf, beta_band, axis=0))

    # Compute entropy
    signal_entropy = np.mean(compute_entropy(window, axis=0))

    # Rule-based classification
    if delta_power > theta_power and signal_entropy < 2.5:
        return np.array([0.8, 0.1, 0.1])  # Likely healthy control
    elif alpha_power > beta_power and signal_entropy >= 2.5:
        return np.array([0.1, 0.7, 0.2])  # Likely frontotemporal dementia
    else:
        return np.array([0.1, 0.2, 0.7])  # Likely Alzheimer's disease
