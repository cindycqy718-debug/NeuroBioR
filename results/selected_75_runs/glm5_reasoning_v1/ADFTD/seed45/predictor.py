import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic EEG source-cohort classifier.
    
    Parameters:
    -----------
    window : np.ndarray, shape [256, 19]
        Standardized EEG window at 256 Hz
        
    Returns:
    --------
    probs : np.ndarray, shape [3]
        Probabilities for [healthy_control, FTD, AD]
    """
    fs = 256
    n_samples, n_channels = window.shape
    
    window = np.asarray(window, dtype=np.float64)
    
    nperseg = min(64, n_samples)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    def band_power(fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.sum(mask) == 0:
            return np.zeros(n_channels)
        return np.mean(psd[mask], axis=0)
    
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    gamma = band_power(30, 45)
    
    total_power = delta + theta + alpha + beta + gamma + 1e-10
    
    rel_delta = float(np.mean(delta / total_power))
    rel_theta = float(np.mean(theta / total_power))
    rel_alpha = float(np.mean(alpha / total_power))
    rel_beta = float(np.mean(beta / total_power))
    
    slow_idx = rel_delta + rel_theta
    alpha_beta_ratio = rel_alpha / (rel_beta + 1e-10)
    
    def hjorth_complexity(x):
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        if var_x < 1e-10 or var_dx < 1e-10:
            return 1.0
        mobility = math.sqrt(var_dx / var_x)
        if mobility < 1e-10 or var_ddx < 1e-10:
            return 1.0
        return math.sqrt(var_ddx / var_dx) / mobility
    
    complexities = [hjorth_complexity(window[:, ch]) for ch in range(n_channels)]
    complexity_mean = float(np.mean(complexities))
    
    def spectral_entropy(psd_ch):
        psd_sum = np.sum(psd_ch)
        if psd_sum < 1e-10:
            return 0.0
        psd_norm = psd_ch / psd_sum
        psd_norm = psd_norm[psd_norm > 1e-10]
        if len(psd_norm) == 0:
            return 0.0
        return float(-np.sum(psd_norm * np.log2(psd_norm)))
    
    entropies = [spectral_entropy(psd[:, ch]) for ch in range(n_channels)]
    entropy_mean = float(np.mean(entropies))
    
    if n_channels > 1:
        corr_matrix = np.corrcoef(window.T)
        mask = ~np.eye(n_channels, dtype=bool)
        mean_corr = float(np.mean(np.abs(corr_matrix[mask])))
    else:
        mean_corr = 0.0
    
    s_healthy = 1.0
    s_ftd = 1.0
    s_ad = 1.0
    
    if slow_idx > 0.55:
        s_ad += 3.0
        s_ftd += 2.0
    elif slow_idx > 0.40:
        s_ad += 2.0
        s_ftd += 1.5
    elif slow_idx > 0.30:
        s_ad += 1.0
        s_ftd += 1.0
    elif slow_idx < 0.20:
        s_healthy += 2.0
    else:
        s_healthy += 0.5
    
    if rel_alpha < 0.12:
        s_ad += 3.0
        s_ftd += 1.0
    elif rel_alpha < 0.20:
        s_ad += 2.0
        s_ftd += 1.5
    elif rel_alpha < 0.28:
        s_ad += 0.5
        s_ftd += 1.0
    elif rel_alpha > 0.40:
        s_healthy += 2.5
    else:
        s_healthy += 1.0
    
    if alpha_beta_ratio < 0.8:
        s_ad += 1.5
    elif alpha_beta_ratio > 1.5:
        s_healthy += 1.0
    
    if complexity_mean < 1.2:
        s_ad += 2.0
        s_ftd += 1.5
    elif complexity_mean < 2.0:
        s_ad += 1.0
        s_ftd += 1.0
    elif complexity_mean > 3.5:
        s_healthy += 2.0
    elif complexity_mean > 2.5:
        s_healthy += 1.0
    
    if entropy_mean < 2.5:
        s_ad += 1.5
        s_ftd += 1.0
    elif entropy_mean < 3.5:
        s_ad += 0.5
        s_ftd += 0.5
    elif entropy_mean > 4.5:
        s_healthy += 1.5
    
    if mean_corr > 0.65:
        s_ad += 1.5
    elif mean_corr > 0.50:
        s_ad += 0.5
        s_ftd += 0.5
    elif mean_corr < 0.25:
        s_healthy += 1.0
    
    theta_alpha_ratio = rel_theta / (rel_alpha + 1e-10)
    if theta_alpha_ratio > 1.5:
        s_ad += 2.0
    elif theta_alpha_ratio > 1.0:
        s_ad += 1.0
        s_ftd += 0.5
    elif theta_alpha_ratio < 0.5:
        s_healthy += 1.0
    
    delta_alpha_ratio = rel_delta / (rel_alpha + 1e-10)
    if delta_alpha_ratio > 1.5:
        s_ad += 2.0
    elif delta_alpha_ratio > 0.8:
        s_ad += 1.0
        s_ftd += 0.5
    elif delta_alpha_ratio < 0.3:
        s_healthy += 1.0
    
    total_score = s_healthy + s_ftd + s_ad
    probs = np.array([s_healthy, s_ftd, s_ad], dtype=np.float64) / total_score
    
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / np.sum(probs)
    
    return probs
