
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def compute_band_powers(signal, fs=256):
        freqs, psd = welch(signal, fs=fs, axis=0, nperseg=fs)
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)], axis=0)
        theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)], axis=0)
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)], axis=0)
        beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)], axis=0)
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs < 50)], axis=0)
        return delta_power, theta_power, alpha_power, beta_power, gamma_power

    def compute_spectral_entropy(signal, fs=256):
        freqs, psd = welch(signal, fs=fs, axis=0, nperseg=fs)
        psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
        return entropy(psd_norm, axis=0)

    delta, theta, alpha, beta, gamma = compute_band_powers(window)
    spectral_entropy = compute_spectral_entropy(window)

    delta_mean = np.mean(delta)
    theta_mean = np.mean(theta)
    alpha_mean = np.mean(alpha)
    beta_mean = np.mean(beta)
    gamma_mean = np.mean(gamma)
    entropy_mean = np.mean(spectral_entropy)

    if delta_mean > theta_mean and entropy_mean < 1.5:
        return np.array([0.7, 0.2, 0.1])
    elif theta_mean > delta_mean and alpha_mean > beta_mean:
        return np.array([0.1, 0.7, 0.2])
    else:
        return np.array([0.1, 0.2, 0.7])
