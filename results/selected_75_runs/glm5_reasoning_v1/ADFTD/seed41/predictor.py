import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Zero-shot EEG classifier for healthy control, FTD, and AD.
    
    Parameters:
        window: np.ndarray [256, 19] - EEG window at 256 Hz
    
    Returns:
        probs: np.ndarray [3] - probabilities for [healthy, FTD, AD]
    """
    fs = 256
    n_chan = window.shape[1]
    
    # Spectral analysis using Welch's method (axis=0 for time axis)
    freqs, pxx = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # Compute band powers
    def band_power(low, high):
        idx = (freqs >= low) & (freqs <= high)
        return np.mean(pxx[idx], axis=0)
    
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    
    # Total power and relative powers
    total = delta + theta + alpha + beta + 1e-12
    rel_delta = delta / total
    rel_theta = theta / total
    rel_alpha = alpha / total
    rel_beta = beta / total
    
    # Global mean relative powers
    m_delta = np.mean(rel_delta)
    m_theta = np.mean(rel_theta)
    m_alpha = np.mean(rel_alpha)
    m_beta = np.mean(rel_beta)
    
    # Slow wave dominance (key pathology marker)
    slow_power = m_delta + m_theta
    
    # Spectral entropy (complexity measure)
    psd_norm = pxx / (np.sum(pxx, axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
    entropy = entropy / np.log(pxx.shape[0])
    m_entropy = np.mean(entropy)
    
    # Functional connectivity (mean absolute correlation)
    corr_matrix = np.corrcoef(window.T)
    off_diag = ~np.eye(n_chan, dtype=bool)
    m_corr = np.mean(np.abs(corr_matrix[off_diag]))
    
    # Spatial heterogeneity (focal changes indicator for FTD)
    alpha_var = np.std(rel_alpha)
    delta_var = np.std(rel_delta)
    
    # Alpha peak frequency (slowing in AD)
    alpha_idx = (freqs >= 8) & (freqs <= 13)
    alpha_psd = np.mean(pxx[alpha_idx], axis=1)
    alpha_peak = freqs[alpha_idx][np.argmax(alpha_psd)]
    
    # Theta/Alpha ratio (key AD marker)
    theta_alpha_ratio = m_theta / (m_alpha + 1e-12)
    
    # === Feature-based scoring based on clinical EEG characteristics ===
    
    # Healthy control: preserved alpha, high complexity, good connectivity
    healthy = (
        m_alpha * 4.0 +
        m_entropy * 3.0 +
        m_corr * 2.0 +
        (1.0 - slow_power) * 2.0 +
        max(0.0, alpha_peak - 9.0) * 0.2
    )
    
    # FTD: focal frontal-temporal changes, spatial heterogeneity
    ftd = (
        alpha_var * 5.0 +
        delta_var * 3.0 +
        m_delta * 2.5 +
        m_theta * 1.5 +
        (1.0 - m_corr) * 0.8
    )
    
    # AD: diffuse slowing, reduced alpha, reduced connectivity
    ad = (
        slow_power * 3.5 +
        (1.0 - m_alpha) * 3.0 +
        (1.0 - m_entropy) * 2.5 +
        (1.0 - m_corr) * 1.5 +
        theta_alpha_ratio * 1.5 +
        max(0.0, 10.0 - alpha_peak) * 0.25
    )
    
    # Softmax normalization
    scores = np.array([healthy, ftd, ad], dtype=np.float64)
    max_s = np.max(scores)
    exp_scores = np.exp(scores - max_s)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs
