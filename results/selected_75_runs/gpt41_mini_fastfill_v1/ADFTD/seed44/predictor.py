import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs < band[1])
    band_power = np.sum(psd[idx], axis=0)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1, total_power)
    return band_power / total_power

def spectral_entropy(psd, base=2):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    psd_norm = np.where(psd_norm == 0, 1e-12, psd_norm)
    se = -np.sum(psd_norm * np.log(psd_norm) / np.log(base), axis=0)
    return se

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=256, noverlap=128)
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 45)
    delta_power = bandpower(psd, freqs, delta)
    theta_power = bandpower(psd, freqs, theta)
    alpha_power = bandpower(psd, freqs, alpha)
    beta_power = bandpower(psd, freqs, beta)
    gamma_power = bandpower(psd, freqs, gamma)
    delta_mean = np.mean(delta_power)
    theta_mean = np.mean(theta_power)
    alpha_mean = np.mean(alpha_power)
    beta_mean = np.mean(beta_power)
    gamma_mean = np.mean(gamma_power)
    se = np.mean(spectral_entropy(psd))
    temporal_std = np.mean(np.std(window, axis=0))
    corr_matrix = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[iu])
    score_healthy = (alpha_mean * 2 + beta_mean * 1.5 - delta_mean - theta_mean + (1 - se) + mean_corr)
    score_ftd = (theta_mean * 2 + delta_mean * 1.5 - alpha_mean - beta_mean + (1 - se)*0.5 + (1 - mean_corr)*0.5)
    score_ad = (delta_mean * 2 + theta_mean * 1.5 - alpha_mean - beta_mean + se + (1 - mean_corr))
    scores = np.array([score_healthy, score_ftd, score_ad])
    scores = scores - np.min(scores)
    if np.sum(scores) == 0:
        probs = np.array([1/3, 1/3, 1/3])
    else:
        probs = scores / np.sum(scores)
    return probs
