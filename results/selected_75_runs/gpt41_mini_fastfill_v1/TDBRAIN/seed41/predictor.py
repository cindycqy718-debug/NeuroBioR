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
    delta = (1, 4)
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
    total_power = delta_mean + theta_mean + alpha_mean + beta_mean
    beta_rel = beta_mean / total_power if total_power > 0 else 0
    spec_ent = np.mean(spectral_entropy(psd, freqs))
    temporal_std = np.mean(np.std(window, axis=0))
    corr = np.corrcoef(window.T)
    upper_tri_idx = np.triu_indices_from(corr, k=1)
    mean_corr = np.mean(corr[upper_tri_idx])
    w_beta = 3.0
    w_entropy = 2.5
    w_corr = 1.5
    score = w_beta * beta_rel - w_entropy * spec_ent + w_corr * mean_corr
    threshold = 0.0
    p_parkinson = 1 / (1 + np.exp(-5 * (score - threshold)))
    p_healthy = 1 - p_parkinson
    p_parkinson = max(0.0, min(1.0, p_parkinson))
    p_healthy = max(0.0, min(1.0, p_healthy))
    s = p_healthy + p_parkinson
    if s > 0:
        p_healthy /= s
        p_parkinson /= s
    else:
        p_healthy = 0.5
        p_parkinson = 0.5
    return np.array([p_healthy, p_parkinson])
