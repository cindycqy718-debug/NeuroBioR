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

    spec_ent = np.mean(spectral_entropy(psd, freqs))
    temporal_var = np.mean(np.std(window, axis=0))
    corr_mat = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_mat, k=1)
    mean_corr = np.mean(corr_mat[iu])

    healthy_score = (alpha_mean + beta_mean) * spec_ent * mean_corr
    ftd_score = (alpha_mean * 0.8 + beta_mean * 0.9) * (1 - spec_ent) * (delta_mean * 0.5 + mean_corr * 0.5)
    ad_score = (delta_mean + theta_mean) * (1 - alpha_mean) * (1 - beta_mean) * (1 - spec_ent)

    scores = np.array([healthy_score, ftd_score, ad_score])
    scores = np.clip(scores, 0, None)
    if np.sum(scores) == 0:
        probs = np.array([1/3, 1/3, 1/3])
    else:
        probs = scores / np.sum(scores)

    return probs.astype(float)
