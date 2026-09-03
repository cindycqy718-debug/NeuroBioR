import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window: np.ndarray) -> np.ndarray:
    fs = 256
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    f, psd = welch(window, fs=fs, axis=0, nperseg=128)
    total_power = np.trapz(psd, f, axis=0) + 1e-12
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(f >= fmin, f <= fmax)
        band_power = np.trapz(psd[idx, :], f[idx], axis=0)
        band_powers[band] = band_power / total_power
    avg_band_powers = {band: np.mean(power) for band, power in band_powers.items()}
    alpha_rel = avg_band_powers['alpha']
    theta_rel = avg_band_powers['theta']
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-12)
    spectral_entropy = np.mean(entropy(psd_norm, base=2, axis=0))
    temporal_std = np.mean(np.std(window, axis=0))
    corr_matrix = np.corrcoef(window.T)
    upper_tri_idx = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = np.mean(corr_matrix[upper_tri_idx])
    score = 0
    score += 2 * theta_rel
    score -= 2 * alpha_rel
    score -= 1.5 * spectral_entropy
    score -= 1.0 * temporal_std
    score -= 1.0 * mean_corr
    prob_alz = 1 / (1 + np.exp(-score))
    prob_healthy = 1 - prob_alz
    return np.array([prob_healthy, prob_alz])
