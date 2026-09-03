import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower_psd(psd, freqs, band):
    freq_mask = (freqs >= band[0]) & (freqs < band[1])
    band_power = np.sum(psd[freq_mask], axis=0)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1e-10, total_power)
    return band_power / total_power

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    psd_norm = np.clip(psd_norm, 1e-12, None)
    ent = -np.sum(psd_norm * np.log2(psd_norm), axis=0)
    ent /= np.log2(psd.shape[0])
    return ent

def predict(window):
    fs = 256
    nperseg = 256
    delta = (1,4)
    theta = (4,8)
    alpha = (8,13)
    beta = (13,30)
    gamma = (30,45)

    freqs, psd = welch(window, fs=fs, nperseg=nperseg, axis=0)

    delta_power = bandpower_psd(psd, freqs, delta)
    theta_power = bandpower_psd(psd, freqs, theta)
    alpha_power = bandpower_psd(psd, freqs, alpha)
    beta_power = bandpower_psd(psd, freqs, beta)
    gamma_power = bandpower_psd(psd, freqs, gamma)

    delta_mean = np.mean(delta_power)
    theta_mean = np.mean(theta_power)
    alpha_mean = np.mean(alpha_power)
    beta_mean = np.mean(beta_power)
    gamma_mean = np.mean(gamma_power)

    spec_ent = np.mean(spectral_entropy(psd, freqs))

    temporal_std = np.mean(np.std(window, axis=0))

    corr_matrix = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[iu])

    score = (
        -3.0 * alpha_mean +
        +2.5 * theta_mean +
        -2.0 * beta_mean +
        -4.0 * spec_ent +
        -1.5 * mean_corr
    )

    prob_parkinson = 1 / (1 + np.exp(-score))
    prob_healthy = 1 - prob_parkinson

    probs = np.array([prob_healthy, prob_parkinson])
    probs = np.clip(probs, 0, 1)
    probs /= np.sum(probs)

    return probs
