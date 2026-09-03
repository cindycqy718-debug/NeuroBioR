import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.sum(psd[idx_band], axis=0)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1, total_power)
    return bp / total_power

def spectral_entropy(psd, axis=0):
    psd_norm = psd / np.sum(psd, axis=axis, keepdims=True)
    psd_norm = np.clip(psd_norm, 1e-12, None)
    return entropy(psd_norm, base=2, axis=axis)

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, nperseg=128, axis=0)
    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 45)
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
    spec_ent = np.mean(spectral_entropy(psd, axis=0))
    temporal_var = np.mean(np.std(window, axis=0))
    corr_mat = np.corrcoef(window.T)
    iu = np.triu_indices_from(corr_mat, k=1)
    mean_corr = np.mean(corr_mat[iu])
    score_hc = (mean_alpha * 3.0) + (mean_beta * 1.5) + (spec_ent * 2.0) + (mean_corr * 1.0)
    score_ftd = (mean_theta * 3.0) + ((1 - mean_alpha) * 2.5) + ((1 - spec_ent) * 2.0) + ((1 - mean_corr) * 1.5)
    score_ad = (mean_delta * 3.5) + ((1 - mean_alpha) * 2.0) + ((1 - mean_beta) * 1.5) + ((1 - spec_ent) * 2.0) + (mean_corr * 1.0)
    scores = np.array([score_hc, score_ftd, score_ad])
    scores = np.clip(scores, 0, None)
    if np.sum(scores) == 0:
        probs = np.array([1/3, 1/3, 1/3])
    else:
        probs = scores / np.sum(scores)
    return probs
