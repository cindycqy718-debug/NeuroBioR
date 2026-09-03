import numpy as np
from scipy import signal
import math

def predict(window):
    """
    Deterministic zero-shot EEG classifier for healthy vs Parkinson's source cohort.
    
    Uses physiologically justified features based on PD EEG literature:
    - Beta/alpha power ratio (elevated in PD due to basal ganglia dysfunction)
    - Relative band powers (alpha often reduced, beta elevated in PD)
    - Spectral entropy (altered spectral distribution in PD)
    - Hjorth complexity (signal complexity changes in PD)
    - Cross-channel synchronization (connectivity alterations in PD)
    
    Parameters:
    -----------
    window : np.ndarray
        EEG window of shape [256, 33] at 256 Hz (1 second of data)
        
    Returns:
    --------
    np.ndarray
        Probabilities [p_healthy, p_parkinson] that sum to 1
    """
    fs = 256
    window = np.asarray(window, dtype=np.float64)
    
    # Compute PSD using Welch's method (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, nperseg=64, axis=0)
    
    # Helper function for band power
    def band_power(low, high):
        idx = (freqs >= low) & (freqs <= high)
        return np.mean(psd[idx, :], axis=0)
    
    # Compute absolute band powers for relevant frequency bands
    delta = band_power(1, 4)      # Delta: 1-4 Hz
    theta = band_power(4, 8)      # Theta: 4-8 Hz
    alpha = band_power(8, 13)     # Alpha: 8-13 Hz
    beta = band_power(13, 30)     # Beta: 13-30 Hz
    
    # Total power across bands
    total = delta + theta + alpha + beta + 1e-10
    
    # Relative band powers (mean across channels)
    rel_alpha = np.mean(alpha / total)
    rel_beta = np.mean(beta / total)
    rel_theta = np.mean(theta / total)
    
    # Beta/alpha ratio - key biomarker elevated in PD
    beta_alpha_ratio = rel_beta / (rel_alpha + 1e-10)
    
    # Spectral entropy - measures spectral distribution complexity
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    spec_entropy = np.mean(-np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0))
    
    # Hjorth complexity - measures signal complexity
    diff1 = np.diff(window, axis=0)
    diff2 = np.diff(diff1, axis=0)
    var_sig = np.var(window, axis=0) + 1e-10
    var_d1 = np.var(diff1, axis=0) + 1e-10
    var_d2 = np.var(diff2, axis=0) + 1e-10
    mobility = np.sqrt(var_d1 / var_sig)
    mobility_d1 = np.sqrt(var_d2 / var_d1)
    complexity = np.mean(mobility_d1 / (mobility + 1e-10))
    
    # Cross-channel correlation - functional connectivity measure
    # Handle edge case where variance is zero
    std_vals = np.std(window, axis=0)
    valid_channels = std_vals > 1e-10
    if np.sum(valid_channels) > 1:
        valid_window = window[:, valid_channels]
        corr = np.corrcoef(valid_window.T)
        if corr.ndim == 2 and corr.shape[0] > 1:
            triu = corr[np.triu_indices(corr.shape[0], k=1)]
            mean_corr = np.mean(np.abs(triu))
        else:
            mean_corr = 0.3
    else:
        mean_corr = 0.3
    
    # Signal variability
    signal_var = np.mean(np.var(window, axis=0))
    
    # PD score based on established EEG literature:
    # - Beta/alpha ratio elevated in PD (positive weight)
    # - Alpha power often reduced in PD (negative weight for alpha)
    # - Spectral entropy alterations
    # - Complexity changes
    # - Connectivity alterations
    
    pd_score = 0.0
    
    # Beta/alpha ratio: typical ~0.3-0.5, elevated in PD
    pd_score += 1.8 * (beta_alpha_ratio - 0.40)
    
    # Alpha power: typical ~0.25-0.35, often reduced in PD
    pd_score -= 2.2 * (rel_alpha - 0.30)
    
    # Theta power: may be altered in PD
    pd_score += 0.5 * (rel_theta - 0.20)
    
    # Spectral entropy: typical ~3-4, altered in PD
    pd_score += 0.4 * (spec_entropy - 3.5)
    
    # Complexity: typical ~1.0, altered in PD
    pd_score += 0.3 * (complexity - 1.0)
    
    # Correlation: typical ~0.2-0.4, altered connectivity in PD
    pd_score += 0.6 * (mean_corr - 0.30)
    
    # Signal variance normalization
    log_var = np.log(signal_var + 1e-10)
    pd_score += 0.1 * (log_var - 5.0)
    
    # Convert to probability using logistic function
    pd_prob = 1.0 / (1.0 + np.exp(-pd_score))
    
    # Ensure valid probabilities in [0.01, 0.99]
    pd_prob = float(np.clip(pd_prob, 0.01, 0.99))
    healthy_prob = 1.0 - pd_prob
    
    return np.array([healthy_prob, pd_prob], dtype=np.float64)
