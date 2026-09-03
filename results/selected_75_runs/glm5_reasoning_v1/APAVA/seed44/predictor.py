import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG source-cohort classifier.
    
    Classifies based on established AD EEG markers:
    - EEG slowing (increased delta/theta, decreased alpha)
    - Reduced spectral entropy
    - Decreased cross-channel correlation
    - Reduced signal complexity (Hjorth parameters)
    
    Parameters:
    -----------
    window : np.ndarray
        EEG window of shape [256, 16] at 256 Hz
        
    Returns:
    --------
    np.ndarray
        Probabilities [p_healthy, p_alzheimer]
    """
    fs = 256
    n_samples, n_channels = window.shape
    
    # Compute PSD using Welch method (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=64)
    psd_avg = np.mean(psd, axis=1)
    
    # Helper: compute band power
    def band_power(f, p, f_low, f_high):
        mask = (f >= f_low) & (f <= f_high)
        return np.sum(p[mask])
    
    total_power = np.sum(psd_avg) + 1e-10
    
    # Relative band powers
    delta = band_power(freqs, psd_avg, 0.5, 4) / total_power
    theta = band_power(freqs, psd_avg, 4, 8) / total_power
    alpha = band_power(freqs, psd_avg, 8, 13) / total_power
    beta = band_power(freqs, psd_avg, 13, 30) / total_power
    
    # Feature 1: Slowing index - hallmark of AD
    # Higher in AD (shift from fast to slow rhythms)
    slowing_index = (delta + theta) / (alpha + beta + 1e-10)
    
    # Feature 2: Alpha power - reduced in AD
    alpha_ratio = alpha
    
    # Feature 3: Spectral entropy - reduced in AD (less complex spectrum)
    psd_norm = psd_avg / total_power
    psd_norm = psd_norm[psd_norm > 1e-12]
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    
    # Feature 4: Cross-channel correlation - reduced functional connectivity in AD
    if n_channels > 1:
        corr_matrix = np.corrcoef(window.T)
        mask = ~np.eye(n_channels, dtype=bool)
        avg_corr = np.mean(np.abs(corr_matrix[mask]))
    else:
        avg_corr = 0.5
    
    # Feature 5: Hjorth complexity - reduced in AD
    def hjorth_complexity(x):
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        if var_x > 1e-12 and var_dx > 1e-12:
            mobility = math.sqrt(var_dx / var_x)
            mobility_dx = math.sqrt(var_ddx / var_dx)
            return mobility_dx / mobility if mobility > 1e-12 else 1.0
        return 1.0
    
    complexities = [hjorth_complexity(window[:, ch]) for ch in range(n_channels)]
    avg_complexity = np.mean(complexities)
    
    # Feature 6: Theta/alpha ratio - elevated in AD
    theta_alpha_ratio = theta / (alpha + 1e-10)
    
    # Normalize features using sigmoid functions with physiologically-based thresholds
    # Slowing index: typical range 0.5-3, higher = more AD
    f1 = 1.0 / (1.0 + math.exp(-2.0 * (slowing_index - 1.5)))
    
    # Alpha ratio: typical range 0.1-0.5, lower = more AD
    f2 = 1.0 / (1.0 + math.exp(12.0 * (alpha_ratio - 0.25)))
    
    # Spectral entropy: typical range 2-6 bits, lower = more AD
    f3 = 1.0 / (1.0 + math.exp(1.2 * (spec_entropy - 4.0)))
    
    # Correlation: typical range 0.2-0.9, lower = more AD
    f4 = 1.0 / (1.0 + math.exp(6.0 * (avg_corr - 0.55)))
    
    # Complexity: typical range 0.5-2.0, lower = more AD
    f5 = 1.0 / (1.0 + math.exp(2.5 * (avg_complexity - 1.25)))
    
    # Theta/alpha ratio: typical range 0.3-2.0, higher = more AD
    f6 = 1.0 / (1.0 + math.exp(-2.0 * (theta_alpha_ratio - 1.0)))
    
    # Weighted combination based on clinical significance
    # Slowing and alpha changes are most robust AD markers
    weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    features = [f1, f2, f3, f4, f5, f6]
    
    ad_score = sum(w * f for w, f in zip(weights, features))
    
    # Ensure valid probability bounds
    ad_prob = float(np.clip(ad_score, 0.01, 0.99))
    healthy_prob = 1.0 - ad_prob
    
    return np.array([healthy_prob, ad_prob], dtype=np.float64)
