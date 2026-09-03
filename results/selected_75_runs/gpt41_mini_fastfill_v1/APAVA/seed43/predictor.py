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
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 45)
    delta_p = bandpower(psd, freqs, delta)
    theta_p = bandpower(psd, freqs, theta)
    alpha_p = bandpower(psd, freqs, alpha)
    beta_p = bandpower(psd, freqs, beta)
    gamma_p = bandpower(psd, freqs, gamma)
    delta_mean = np.mean(delta_p)
    theta_mean = np.mean(theta_p)
    alpha_mean = np.mean(alpha_p)
    beta_mean = np.mean(beta_p)
    gamma_mean = np.mean(gamma_p)
    total_power = delta_mean + theta_mean + alpha_mean + beta_mean + gamma_mean
    alpha_rel = alpha_mean / total_power if total_power > 0 else 0
    spec_ent = np.mean(spectral_entropy(psd, freqs))
    temporal_var = np.mean(np.std(window, axis=0))
    corr = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr, k=1)
    mean_corr = np.mean(corr[iu])
    score = alpha_rel - spec_ent * 0.3 + mean_corr * 0.1
    if score > 0.05:
        probs = np.array([0.9, 0.1])
    else:
        probs = np.array([0.1, 0.9])
    probs = probs / np.sum(probs)
    return probs
