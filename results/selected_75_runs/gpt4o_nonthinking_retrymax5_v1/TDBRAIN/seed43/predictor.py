
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def band_power(signal, fs, band):
        f, Pxx = welch(signal, fs=fs, axis=0, nperseg=128)
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        return np.sum(Pxx[idx_band], axis=0)

    def spectral_entropy(signal, fs):
        f, Pxx = welch(signal, fs=fs, axis=0, nperseg=128)
        Pxx_norm = Pxx / np.sum(Pxx, axis=0, keepdims=True)
        return entropy(Pxx_norm, axis=0)

    fs = 256  # Sampling frequency
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    delta_power = np.mean(band_power(window, fs, delta_band))
    theta_power = np.mean(band_power(window, fs, theta_band))
    alpha_power = np.mean(band_power(window, fs, alpha_band))
    beta_power = np.mean(band_power(window, fs, beta_band))
    total_power = delta_power + theta_power + alpha_power + beta_power

    spectral_entropy_mean = np.mean(spectral_entropy(window, fs))

    # Decision rule based on band power ratios and spectral entropy
    if total_power == 0:
        return np.array([0.5, 0.5])  # Handle edge case of zero power

    alpha_beta_ratio = alpha_power / (beta_power + 1e-8)  # Avoid division by zero
    delta_theta_ratio = delta_power / (theta_power + 1e-8)

    if alpha_beta_ratio > 0.8 and spectral_entropy_mean < 2.5:
        return np.array([0.8, 0.2])  # Likely healthy_source_cohort
    elif delta_theta_ratio > 1.2 and spectral_entropy_mean > 3.0:
        return np.array([0.2, 0.8])  # Likely parkinson_source_cohort
    else:
        return np.array([0.5, 0.5])  # Uncertain case
