import numpy as np
import scipy.signal
import math

def predict(window: np.ndarray) -> np.ndarray:
    fs = 256.0
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    f, Pxx = scipy.signal.welch(window, fs=fs, axis=0, nperseg=128, noverlap=64)
    total_power = np.trapz(Pxx, f, axis=0) + 1e-12
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(f >= fmin, f <= fmax)
        band_power = np.trapz(Pxx[idx, :], f[idx], axis=0)
        band_powers[band] = band_power / total_power
    avg_band_powers = {band: np.mean(power) for band, power in band_powers.items()}
    beta_power = avg_band_powers['beta']
    alpha_power = avg_band_powers['alpha']
    analytic_signal = scipy.signal.hilbert(window, axis=0)
    envelope = np.abs(analytic_signal)
    temporal_var = np.mean(np.std(envelope, axis=0))
    def spectral_entropy(psd, freqs):
        psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
        psd_norm = np.clip(psd_norm, 1e-12, None)
        se = -np.sum(psd_norm * np.log2(psd_norm), axis=0)
        se /= np.log2(psd_norm.shape[0])
        return se
    spec_entropy = np.mean(spectral_entropy(Pxx, f))
    corr_matrix = np.corrcoef(window.T)
    abs_corr = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    mean_abs_corr = np.mean(abs_corr)
    beta_norm = beta_power / 0.3
    alpha_norm = alpha_power / 0.4
    temporal_var_norm = temporal_var / 1.5
    spec_entropy_norm = (spec_entropy - 0.5) / 0.5
    mean_abs_corr_norm = mean_abs_corr
    parkinson_score = (
        + beta_norm * 1.5
        - alpha_norm * 1.0
        - temporal_var_norm * 1.0
        - spec_entropy_norm * 1.0
        + mean_abs_corr_norm * 1.0
    )
    score_shifted = parkinson_score - 0.5
    p_parkinson = 1 / (1 + math.exp(-score_shifted))
    p_healthy = 1 - p_parkinson
    return np.array([p_healthy, p_parkinson], dtype=np.float64)
