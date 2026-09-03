import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    freq_mask = (freqs >= band[0]) & (freqs < band[1])
    band_power = np.sum(psd[freq_mask], axis=0)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1e-10, total_power)
    return band_power / total_power

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    psd_norm = np.clip(psd_norm, 1e-12, None)
    ent = -np.sum(psd_norm * np.log2(psd_norm), axis=0)
    max_ent = np.log2(psd.shape[0])
    return ent / max_ent

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=128, noverlap=64)
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
    spec_ent = np.mean(spectral_entropy(psd, freqs))
    temporal_var = np.mean(np.std(window, axis=0))
    corr = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr, k=1)
    mean_abs_corr = np.mean(np.abs(corr[iu]))
    score_hc = (
        alpha_mean * 3.0 +
        beta_mean * 1.5 +
        spec_ent * 2.0 +
        (1 - delta_mean - theta_mean) * 1.0 +
        mean_abs_corr * 1.0
    )
    score_ftd = (
        theta_mean * 3.0 +
        alpha_mean * 1.0 +
        spec_ent * 1.0 +
        mean_abs_corr * 1.5 +
        temporal_var * 0.5
    )
    score_ad = (
        delta_mean * 3.0 +
        theta_mean * 2.0 +
        (1 - alpha_mean) * 2.0 +
        (1 - beta_mean) * 1.0 +
        (1 - spec_ent) * 3.0
    )
    scores = np.array([score_hc, score_ftd, score_ad])
    scores = np.clip(scores, 0, None)
    total = np.sum(scores)
    if total == 0:
        probs = np.array([1/3, 1/3, 1/3])
    else:
        probs = scores / total
    return probs
