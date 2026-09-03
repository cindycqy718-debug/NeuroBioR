
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def bandpower(data, sf, band, axis=0):
        band = np.asarray(band)
        freqs, psd = welch(data, sf, axis=axis)
        band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.sum(psd[band_idx], axis=0)

    sf = 256
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    delta_power = bandpower(window, sf, delta_band, axis=0)
    theta_power = bandpower(window, sf, theta_band, axis=0)
    alpha_power = bandpower(window, sf, alpha_band, axis=0)
    beta_power = bandpower(window, sf, beta_band, axis=0)

    total_power = delta_power + theta_power + alpha_power + beta_power
    delta_ratio = np.mean(delta_power / total_power)
    theta_ratio = np.mean(theta_power / total_power)
    alpha_ratio = np.mean(alpha_power / total_power)
    beta_ratio = np.mean(beta_power / total_power)

    freqs, psd = welch(window, sf, axis=0)
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    spectral_entropy = np.mean(entropy(psd_norm, axis=0))

    if delta_ratio > 0.4 or spectral_entropy < 2.5:
        return np.array([0.9, 0.1])
    else:
        return np.array([0.1, 0.9])
