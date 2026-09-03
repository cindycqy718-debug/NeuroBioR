
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def compute_band_power(signal, fs, band):
        f, Pxx = welch(signal, fs=fs, axis=0, nperseg=fs)
        band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])], axis=0)
        return band_power

    def compute_entropy(signal):
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        return entropy(hist)

    # Sampling frequency
    fs = 256

    # Frequency bands
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    # Compute band powers
    delta_power = compute_band_power(window, fs, delta_band)
    theta_power = compute_band_power(window, fs, theta_band)
    alpha_power = compute_band_power(window, fs, alpha_band)
    beta_power = compute_band_power(window, fs, beta_band)

    # Compute total power and relative band powers
    total_power = delta_power + theta_power + alpha_power + beta_power
    relative_delta = np.mean(delta_power / total_power)
    relative_theta = np.mean(theta_power / total_power)
    relative_alpha = np.mean(alpha_power / total_power)
    relative_beta = np.mean(beta_power / total_power)

    # Compute entropy
    temporal_entropy = np.mean([compute_entropy(window[:, ch]) for ch in range(window.shape[1])])

    # Decision rule based on features
    if relative_delta > 0.4 or temporal_entropy < 2.5:
        return np.array([0.9, 0.1])  # Likely healthy_source_cohort
    else:
        return np.array([0.1, 0.9])  # Likely alzheimer_source_cohort
