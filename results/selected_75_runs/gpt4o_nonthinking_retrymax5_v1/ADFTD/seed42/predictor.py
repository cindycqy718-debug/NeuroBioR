
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    if window.shape != (256, 19):
        raise ValueError("Input window must have shape [256, 19].")
    
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    fs = 256
    
    band_power_ratios = []
    for channel in range(window.shape[1]):
        f, psd = welch(window[:, channel], fs=fs, axis=0)
        total_power = np.sum(psd)
        delta_power = np.sum(psd[(f >= delta_band[0]) & (f < delta_band[1])])
        theta_power = np.sum(psd[(f >= theta_band[0]) & (f < theta_band[1])])
        alpha_power = np.sum(psd[(f >= alpha_band[0]) & (f < alpha_band[1])])
        beta_power = np.sum(psd[(f >= beta_band[0]) & (f < beta_band[1])])
        delta_ratio = delta_power / total_power
        theta_ratio = theta_power / total_power
        alpha_ratio = alpha_power / total_power
        beta_ratio = beta_power / total_power
        band_power_ratios.append([delta_ratio, theta_ratio, alpha_ratio, beta_ratio])
    
    band_power_ratios = np.array(band_power_ratios)
    mean_ratios = np.mean(band_power_ratios, axis=0)
    var_ratios = np.var(band_power_ratios, axis=0)
    spectral_entropies = []
    for channel in range(window.shape[1]):
        f, psd = welch(window[:, channel], fs=fs, axis=0)
        psd_norm = psd / np.sum(psd)
        spectral_entropies.append(entropy(psd_norm))
    mean_entropy = np.mean(spectral_entropies)
    features = np.concatenate([mean_ratios, var_ratios, [mean_entropy]])
    if features[0] > 0.4:
        return np.array([0.7, 0.2, 0.1])
    elif features[2] > 0.3:
        return np.array([0.1, 0.7, 0.2])
    else:
        return np.array([0.1, 0.2, 0.7])
