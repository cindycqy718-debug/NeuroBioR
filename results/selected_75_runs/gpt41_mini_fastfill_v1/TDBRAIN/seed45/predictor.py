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
    bp_delta = bandpower(psd, freqs, delta)
    bp_theta = bandpower(psd, freqs, theta)
    bp_alpha = bandpower(psd, freqs, alpha)
    bp_beta = bandpower(psd, freqs, beta)
    bp_gamma = bandpower(psd, freqs, gamma)
    total_power = bp_delta + bp_theta + bp_alpha + bp_beta + bp_gamma + 1e-12
    rel_beta = np.mean(bp_beta / total_power)
    spec_ent = spectral_entropy(psd, freqs)
    mean_spec_ent = np.mean(spec_ent)
    corr = np.corrcoef(window.T)
    upper_tri_idx = np.triu_indices_from(corr, k=1)
    mean_corr = np.mean(corr[upper_tri_idx])
    score = 0.6 * rel_beta - 0.3 * mean_spec_ent - 0.1 * mean_corr
    p1 = 1 / (1 + np.exp(-10*(score - 0.1)))
    p0 = 1 - p1
    p0 = float(np.clip(p0, 0, 1))
    p1 = float(np.clip(p1, 0, 1))
    s = p0 + p1
    p0 /= s
    p1 /= s
    return np.array([p0, p1], dtype=float)
