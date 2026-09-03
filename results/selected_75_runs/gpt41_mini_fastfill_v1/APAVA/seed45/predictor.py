import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window: np.ndarray) -> np.ndarray:
    fs = 256
    nperseg = 128
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    f, Pxx = welch(window, fs=fs, nperseg=nperseg, axis=0)
    total_power = np.trapezoid(Pxx, f, axis=0)
    band_powers = []
    for band in bands.values():
        fmin, fmax = band
        idx = np.logical_and(f >= fmin, f <= fmax)
        band_power = np.trapezoid(Pxx[idx, :], f[idx], axis=0) / (total_power + 1e-12)
        band_powers.append(np.mean(band_power))
    band_powers = np.array(band_powers)
    psd_norm = Pxx / (np.sum(Pxx, axis=0, keepdims=True) + 1e-12)
    spectral_entropy = np.mean(entropy(psd_norm, base=2, axis=0))
    temporal_std = np.mean(np.std(window, axis=0))
    complexity = np.mean(np.var(window, axis=0) / (np.mean(np.abs(window), axis=0) + 1e-12))
    corr_matrix = np.corrcoef(window.T)
    corr_vals = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    mean_abs_corr = np.mean(np.abs(corr_vals))
    features = np.hstack([band_powers, spectral_entropy, temporal_std, complexity, mean_abs_corr])
    weights = np.array([+1.0, +0.8, -1.5, 0.0, 0.0, -1.0, +0.5, -0.5, -1.0])
    score = np.dot(features, weights)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    p_alzheimer = sigmoid(score)
    p_healthy = 1 - p_alzheimer
    return np.array([p_healthy, p_alzheimer])
