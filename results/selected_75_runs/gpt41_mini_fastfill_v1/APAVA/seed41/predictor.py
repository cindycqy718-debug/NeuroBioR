import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from math import exp

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx], axis=0)

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    return entropy(psd_norm, base=2, axis=0)

def predict(window):
    fs = 256
    nperseg = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=nperseg, scaling='density')
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)
    delta_power = bandpower(psd, freqs, delta)
    theta_power = bandpower(psd, freqs, theta)
    alpha_power = bandpower(psd, freqs, alpha)
    beta_power = bandpower(psd, freqs, beta)
    delta_mean = np.mean(delta_power)
    theta_mean = np.mean(theta_power)
    alpha_mean = np.mean(alpha_power)
    beta_mean = np.mean(beta_power)
    total_power = delta_mean + theta_mean + alpha_mean + beta_mean + 1e-12
    alpha_rel = alpha_mean / total_power
    spec_ent = np.mean(spectral_entropy(psd, freqs))
    temporal_var = np.mean(np.std(window, axis=0))
    corr_matrix = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[iu])
    score = 0
    score += (0.5 - alpha_rel) * 2.0
    score += (spec_ent - 2.0) * 1.5
    score += (0.5 - mean_corr) * 2.0
    score += (temporal_var - 0.5) * 0.5
    p_alz = 1 / (1 + exp(-score))
    p_healthy = 1 - p_alz
    return np.array([p_healthy, p_alz], dtype=np.float64)
