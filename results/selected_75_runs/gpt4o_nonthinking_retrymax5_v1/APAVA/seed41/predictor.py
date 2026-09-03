
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def band_power(signal, fs, band):
        f, Pxx = welch(signal, fs=fs, axis=0)
        band_mask = (f >= band[0]) & (f <= band[1])
        return np.sum(Pxx[band_mask], axis=0)
    
    fs = 256  # Sampling frequency
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)
    
    # Compute band powers
    delta_power = band_power(window, fs, delta_band)
    theta_power = band_power(window, fs, theta_band)
    alpha_power = band_power(window, fs, alpha_band)
    beta_power = band_power(window, fs, beta_band)
    
    # Normalize band powers
    total_power = delta_power + theta_power + alpha_power + beta_power
    delta_ratio = np.mean(delta_power / total_power)
    theta_ratio = np.mean(theta_power / total_power)
    alpha_ratio = np.mean(alpha_power / total_power)
    beta_ratio = np.mean(beta_power / total_power)
    
    # Compute spectral entropy
    f, Pxx = welch(window, fs=fs, axis=0)
    Pxx_norm = Pxx / np.sum(Pxx, axis=0, keepdims=True)
    spectral_entropy = np.mean(entropy(Pxx_norm, axis=0))
    
    # Decision rule
    if delta_ratio > 0.4 or spectral_entropy < 2.5:
        return np.array([0.9, 0.1])  # Likely healthy_source_cohort
    else:
        return np.array([0.1, 0.9])  # Likely alzheimer_source_cohort
