import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx], axis=0)

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    return entropy(psd_norm, base=2, axis=0)

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=128, noverlap=64)
    delta = (1,4)
    theta = (4,8)
    alpha = (8,13)
    beta = (13,30)
    gamma = (30,45)
    bp_delta = bandpower(psd, freqs, delta)
    bp_theta = bandpower(psd, freqs, theta)
    bp_alpha = bandpower(psd, freqs, alpha)
    bp_beta = bandpower(psd, freqs, beta)
    bp_gamma = bandpower(psd, freqs, gamma)
    mean_delta = np.mean(bp_delta)
    mean_theta = np.mean(bp_theta)
    mean_alpha = np.mean(bp_alpha)
    mean_beta = np.mean(bp_beta)
    mean_gamma = np.mean(bp_gamma)
    total_power = mean_delta + mean_theta + mean_alpha + mean_beta + mean_gamma
    rel_beta = mean_beta / total_power if total_power > 0 else 0
    spec_ent = np.mean(spectral_entropy(psd, freqs))
    corr_matrix = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[iu])
    rel_beta_norm = (rel_beta - 0.1) / 0.3
    rel_beta_norm = np.clip(rel_beta_norm, 0, 1)
    spec_ent_norm = (5 - spec_ent) / 2
    spec_ent_norm = np.clip(spec_ent_norm, 0, 1)
    mean_corr_norm = (0.9 - mean_corr) / 0.6
    mean_corr_norm = np.clip(mean_corr_norm, 0, 1)
    score = 0.5 * rel_beta_norm + 0.3 * spec_ent_norm + 0.2 * mean_corr_norm
    score = np.clip(score, 0, 1)
    return np.array([1 - score, score])
