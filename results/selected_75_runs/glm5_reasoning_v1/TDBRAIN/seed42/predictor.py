"""
Deterministic zero-shot EEG source-cohort classifier for TDBRAIN dataset.
Classifies healthy_source_cohort (0) vs parkinson_source_cohort (1) based on
physiologically justified spectral and complexity features.
"""

import numpy as np
from scipy import signal
import math


def _compute_band_powers(window, fs=256):
    """Compute relative band powers using Welch's method."""
    n_times, n_ch = window.shape
    nperseg = min(64, n_times)
    
    # Welch PSD estimation - axis=0 is time
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    # Frequency bands (Hz)
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    # Total power per channel
    total_power = np.sum(psd, axis=0, keepdims=True) + 1e-12
    
    rel_powers = {}
    for name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = np.sum(psd[mask, :], axis=0)
        rel_powers[name] = band_power / (total_power[0] + 1e-12)
    
    return rel_powers


def _compute_spectral_entropy(window, fs=256):
    """Compute spectral entropy (normalized) per channel."""
    n_times, n_ch = window.shape
    nperseg = min(64, n_times)
    
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    # Normalize PSD to probability distribution
    psd_sum = np.sum(psd, axis=0, keepdims=True) + 1e-12
    psd_norm = psd / psd_sum
    
    # Spectral entropy
    log_psd = np.log2(psd_norm + 1e-12)
    entropy = -np.sum(psd_norm * log_psd, axis=0)
    
    # Normalize by max entropy (log2 of number of frequency bins)
    max_entropy = math.log2(psd.shape[0])
    return entropy / max_entropy


def _compute_hjorth(window):
    """Compute Hjorth parameters per channel."""
    # Activity: variance of signal
    activity = np.var(window, axis=0) + 1e-12
    
    # First derivative
    d1 = np.diff(window, axis=0)
    var_d1 = np.var(d1, axis=0) + 1e-12
    
    # Mobility: sqrt(var(d1) / var(signal))
    mobility = np.sqrt(var_d1 / activity)
    
    # Second derivative
    d2 = np.diff(d1, axis=0)
    var_d2 = np.var(d2, axis=0) + 1e-12
    
    # Complexity: mobility(d1) / mobility(signal)
    mobility_d1 = np.sqrt(var_d2 / var_d1)
    complexity = mobility_d1 / (mobility + 1e-12)
    
    return activity, mobility, complexity


def _compute_temporal_variability(window):
    """Compute coefficient of variation and RMS amplitude."""
    mean_sig = np.mean(window, axis=0)
    std_sig = np.std(window, axis=0)
    
    # Coefficient of variation
    cv = std_sig / (np.abs(mean_sig) + 1e-12)
    
    # RMS amplitude
    rms = np.sqrt(np.mean(window**2, axis=0))
    
    return cv, rms


def _compute_cross_channel_sync(window):
    """Compute mean absolute correlation across channels."""
    n_ch = window.shape[1]
    
    # Check for constant channels (zero variance)
    std_per_ch = np.std(window, axis=0)
    valid_ch = std_per_ch > 1e-10
    
    if np.sum(valid_ch) < 2:
        return 0.5  # Default neutral value for degenerate cases
    
    # Use only valid channels
    valid_window = window[:, valid_ch]
    
    # Correlation matrix
    corr_matrix = np.corrcoef(valid_window.T)
    
    # Handle any NaN from numerical issues
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Upper triangle (excluding diagonal)
    n_valid = valid_window.shape[1]
    if n_valid < 2:
        return 0.5
    
    upper_idx = np.triu_indices(n_valid, k=1)
    upper_vals = corr_matrix[upper_idx]
    
    # Mean absolute correlation
    mean_abs_corr = np.mean(np.abs(upper_vals))
    
    return float(mean_abs_corr)


def _compute_spectral_edge_frequency(window, fs=256, threshold=0.95):
    """Compute spectral edge frequency (95% power) per channel."""
    n_times, n_ch = window.shape
    nperseg = min(64, n_times)
    
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    
    sef = np.zeros(n_ch)
    for ch in range(n_ch):
        cumsum = np.cumsum(psd[:, ch])
        total = cumsum[-1] + 1e-12
        idx = np.searchsorted(cumsum / total, threshold)
        sef[ch] = freqs[min(idx, len(freqs)-1)]
    
    return sef


def predict(window):
    """
    Predict source-cohort probabilities for a single EEG window.
    
    Parameters
    ----------
    window : ndarray, shape (256, 33)
        Standardized EEG window at 256 Hz.
        Axis 0 is time, axis 1 is channels.
    
    Returns
    -------
    probs : ndarray, shape (2,)
        Probabilities [healthy_source_cohort, parkinson_source_cohort].
        Finite, nonnegative, sum to 1.
    """
    window = np.asarray(window, dtype=np.float64)
    
    # Input validation
    if window.shape != (256, 33):
        raise ValueError(f"Expected shape (256, 33), got {window.shape}")
    if not np.all(np.isfinite(window)):
        raise ValueError("Window contains non-finite values")
    
    # ===== Feature Extraction =====
    
    # 1. Relative band powers
    band_powers = _compute_band_powers(window)
    delta = np.mean(band_powers['delta'])
    theta = np.mean(band_powers['theta'])
    alpha = np.mean(band_powers['alpha'])
    beta = np.mean(band_powers['beta'])
    
    # 2. Spectral entropy (complexity measure)
    spec_entropy = np.mean(_compute_spectral_entropy(window))
    
    # 3. Hjorth complexity
    _, _, complexity = _compute_hjorth(window)
    hjorth_complexity = np.mean(complexity)
    
    # 4. Cross-channel synchronization
    cross_sync = _compute_cross_channel_sync(window)
    
    # 5. Spectral edge frequency
    sef = np.mean(_compute_spectral_edge_frequency(window))
    
    # 6. Slow-to-fast ratio (key PD marker)
    slow_power = delta + theta
    fast_power = alpha + beta + 1e-12
    slow_fast_ratio = slow_power / fast_power
    
    # ===== Decision Rule =====
    # Parkinson's EEG characteristics:
    # - Increased delta/theta (spectral slowing)
    # - Decreased alpha (in some studies)
    # - Reduced complexity/entropy
    # - Altered connectivity
    # - Lower spectral edge frequency
    
    # PD score: positive values indicate PD-like pattern
    pd_score = 0.0
    
    # Spectral slowing marker (strongest evidence)
    # PD: increased slow wave activity
    pd_score += (slow_fast_ratio - 0.5) * 4.0  # Centered around typical ratio
    
    # Spectral entropy reduction in PD
    pd_score += (0.85 - spec_entropy) * 2.0  # Lower entropy = more PD
    
    # Hjorth complexity reduction in PD
    pd_score += (3.0 - hjorth_complexity) * 0.3  # Lower complexity = more PD
    
    # Cross-channel sync changes (PD may show altered connectivity)
    pd_score += (0.3 - cross_sync) * 1.0  # Lower sync tendency toward PD
    
    # Spectral edge frequency (lower in PD due to slowing)
    pd_score += (15.0 - sef) * 0.05  # Lower SEF = more PD
    
    # Convert score to probability using logistic function
    # Scale factor to get reasonable probability range
    logit = pd_score * 1.5
    
    # Clip to prevent extreme values
    logit = np.clip(logit, -10.0, 10.0)
    
    # Logistic sigmoid
    prob_pd = 1.0 / (1.0 + math.exp(-logit))
    
    # Ensure valid probability range
    prob_pd = float(np.clip(prob_pd, 0.001, 0.999))
    prob_healthy = 1.0 - prob_pd
    
    return np.array([prob_healthy, prob_pd], dtype=np.float64)
