
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    if window.shape != (256, 33):
        raise ValueError("Input window must have shape [256, 33].")
    freqs, psd = welch(window, fs=256, axis=0, nperseg=128)
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    def band_power(freqs, psd, band):
        band_mask = (freqs >= band[0]) & (freqs < band[1])
        return np.sum(psd[band_mask, :], axis=0)
    delta_power = band_power(freqs, psd, delta_band)
    theta_power = band_power(freqs, psd, theta_band)
    alpha_power = band_power(freqs, psd, alpha_band)
    beta_power = band_power(freqs, psd, beta_band)
    total_power = delta_power + theta_power + alpha_power + beta_power
    relative_delta = np.mean(delta_power / total_power)
    relative_theta = np.mean(theta_power / total_power)
    relative_alpha = np.mean(alpha_power / total_power)
    relative_beta = np.mean(beta_power / total_power)
    psd_flat = np.mean(psd, axis=1)
    psd_flat /= np.sum(psd_flat)
    spectral_entropy = entropy(psd_flat)
    if relative_alpha > relative_theta and spectral_entropy < 3.5:
        return np.array([0.9, 0.1])
    else:
        return np.array([0.1, 0.9])
