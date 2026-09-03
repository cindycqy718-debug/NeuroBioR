import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG source-cohort classifier.
    
    Classifies EEG windows as healthy (class 0) or Alzheimer's (class 1) source cohort
    based on spectral slowing, reduced complexity, and decreased functional connectivity.
    
    Parameters:
    -----------
    window : ndarray, shape [256, 16]
        EEG window at 256 Hz, axis 0 is time, axis 1 is channels
        
    Returns:
    --------
    probs : ndarray, shape [2]
        Probabilities for [healthy_source_cohort, alzheimer_source_cohort]
    """
    window = np.asarray(window, dtype=np.float64)
    n_samples, n_channels = window.shape
    fs = 256.0
    
    # === Spectral Features ===
    nperseg = min(256, n_samples)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    total_power = np.sum(psd, axis=0)
    total_power = np.maximum(total_power, 1e-20)
    
    # Frequency band masks
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    beta_mask = (freqs >= 12.0) & (freqs <= 30.0)
    
    # Relative band powers
    delta_rel = np.mean(np.sum(psd[delta_mask], axis=0) / total_power)
    theta_rel = np.mean(np.sum(psd[theta_mask], axis=0) / total_power)
    alpha_rel = np.mean(np.sum(psd[alpha_mask], axis=0) / total_power)
    beta_rel = np.mean(np.sum(psd[beta_mask], axis=0) / total_power)
    
    # Slow-to-fast ratio (higher in AD)
    slow_fast_ratio = (delta_rel + theta_rel) / (alpha_rel + beta_rel + 1e-10)
    
    # Delta/alpha ratio (higher in AD)
    delta_alpha_ratio = delta_rel / (alpha_rel + 1e-10)
    
    # Spectral edge frequency at 95% (lower in AD)
    cumsum = np.cumsum(psd, axis=0)
    sef95_vals = np.zeros(n_channels)
    for ch in range(n_channels):
        threshold = 0.95 * cumsum[-1, ch]
        idx = np.searchsorted(cumsum[:, ch], threshold)
        sef95_vals[ch] = freqs[min(idx, len(freqs) - 1)]
    mean_sef95 = np.mean(sef95_vals)
    
    # Mean frequency (lower in AD)
    mean_freq = np.mean(np.sum(freqs[:, None] * psd, axis=0) / (total_power + 1e-20))
    
    # === Hjorth Complexity ===
    diff1 = np.diff(window, axis=0)
    diff2 = np.diff(diff1, axis=0)
    
    var0 = np.var(window, axis=0)
    var1 = np.var(diff1, axis=0)
    var2 = np.var(diff2, axis=0)
    
    mobility1 = np.sqrt(var1 / (var0 + 1e-20))
    mobility2 = np.sqrt(var2 / (var1 + 1e-20))
    complexity = mobility2 / (mobility1 + 1e-20)
    mean_complexity = np.mean(complexity)
    
    # === Cross-Channel Correlation ===
    corr_matrix = np.corrcoef(window.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
    mean_corr = np.mean(np.abs(upper_tri))
    
    # === Feature Normalization ===
    # Each feature normalized to [0, 1], higher = more AD-like
    
    f1 = np.clip((slow_fast_ratio - 0.8) / 1.5, 0.0, 1.0)
    f2 = np.clip((delta_alpha_ratio - 0.4) / 1.2, 0.0, 1.0)
    f3 = np.clip((18.0 - mean_sef95) / 8.0, 0.0, 1.0)
    f4 = np.clip((10.0 - mean_freq) / 5.0, 0.0, 1.0)
    f5 = np.clip((8.0 - mean_complexity) / 5.0, 0.0, 1.0)
    f6 = np.clip((0.45 - mean_corr) / 0.3, 0.0, 1.0)
    
    # === Weighted Combination ===
    weights = np.array([0.22, 0.22, 0.14, 0.14, 0.14, 0.14])
    ad_score = float(np.dot(weights, np.array([f1, f2, f3, f4, f5, f6])))
    
    # === Probability Output ===
    p_ad = 1.0 / (1.0 + math.exp(-5.0 * (ad_score - 0.5)))
    
    probs = np.array([1.0 - p_ad, p_ad], dtype=np.float64)
    return probs
