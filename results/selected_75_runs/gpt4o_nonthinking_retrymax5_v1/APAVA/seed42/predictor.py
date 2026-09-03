
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def compute_band_powers(signal, fs=256):
        freqs, psd = welch(signal, fs=fs, axis=0, nperseg=128)
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)
        delta_power = np.sum(psd[(freqs >= delta_band[0]) & (freqs < delta_band[1])], axis=0)
        theta_power = np.sum(psd[(freqs >= theta_band[0]) & (freqs < theta_band[1])], axis=0)
        alpha_power = np.sum(psd[(freqs >= alpha_band[0]) & (freqs < alpha_band[1])], axis=0)
        beta_power = np.sum(psd[(freqs >= beta_band[0]) & (freqs < beta_band[1])], axis=0)
        total_power = delta_power + theta_power + alpha_power + beta_power
        delta_ratio = delta_power / total_power
        theta_ratio = theta_power / total_power
        alpha_ratio = alpha_power / total_power
        beta_ratio = beta_power / total_power
        return delta_ratio, theta_ratio, alpha_ratio, beta_ratio

    def compute_channel_entropy(window):
        return np.array([entropy(np.abs(window[:, ch])) for ch in range(window.shape[1])])

    delta_ratio, theta_ratio, alpha_ratio, beta_ratio = compute_band_powers(window)
    mean_alpha_ratio = np.mean(alpha_ratio)
    mean_theta_ratio = np.mean(theta_ratio)
    alpha_theta_ratio_variability = np.std(alpha_ratio / theta_ratio)
    channel_entropies = compute_channel_entropy(window)
    mean_entropy = np.mean(channel_entropies)
    if mean_alpha_ratio > 0.2 and mean_entropy > 3.5 and alpha_theta_ratio_variability < 0.5:
        return np.array([0.9, 0.1])
    else:
        return np.array([0.1, 0.9])
    