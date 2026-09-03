import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    freq_mask = (freqs >= band[0]) & (freqs < band[1])
    band_power = np.sum(psd[freq_mask], axis=0)
    total_power = np.sum(psd, axis=0)
    total_power = np.maximum(total_power, 1e-12)
    return band_power / total_power

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    psd_norm = np.clip(psd_norm, 1e-12, None)
    ent = -np.sum(psd_norm * np.log(psd_norm), axis=0)
    ent /= np.log(psd.shape[0])
    return np.mean(ent)

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=256, noverlap=128)
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
    spec_ent = spectral_entropy(psd, freqs)
    temporal_std = np.mean(np.std(window, axis=0))
    corr = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr, k=1)
    mean_abs_corr = np.mean(np.abs(corr[iu]))
    total_band = delta_mean + theta_mean + alpha_mean + beta_mean + gamma_mean + 1e-12
    delta_norm = delta_mean / total_band
    theta_norm = theta_mean / total_band
    alpha_norm = alpha_mean / total_band
    beta_norm = beta_mean / total_band
    gamma_norm = gamma_mean / total_band
    score_0 = (alpha_norm * 3.0 +
               beta_norm * 1.5 +
               spec_ent * 2.0 +
               mean_abs_corr * 1.0 +
               temporal_std * 0.5)
    score_1 = (theta_norm * 3.0 +
               alpha_norm * 0.5 +
               spec_ent * 0.5 +
               mean_abs_corr * 0.3 +
               temporal_std * 0.2)
    score_2 = (delta_norm * 3.0 +
               alpha_norm * 0.5 +
               spec_ent * 0.5 +
               mean_abs_corr * 1.5 +
               temporal_std * 0.3)
    scores = np.array([score_0, score_1, score_2])
    scores = np.maximum(scores, 0)
    total = np.sum(scores)
    if total < 1e-12:
        probs = np.array([1/3, 1/3, 1/3])
    else:
        probs = scores / total
    return probs
