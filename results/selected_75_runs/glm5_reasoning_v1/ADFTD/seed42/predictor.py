
import numpy as np
from scipy import signal
import math

def _band_powers(window, fs=256):
    """Compute relative band powers using Welch PSD."""
    n_times, n_channels = window.shape
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=64, scaling='density')
    
    bands = {'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 
             'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 45.0)}
    
    powers = {}
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        powers[name] = np.mean(psd[mask], axis=0)
    
    total = np.sum(psd, axis=0) + 1e-10
    rel = {name: np.mean(powers[name] / total) for name in powers}
    return rel, freqs, psd

def _spectral_features(freqs, psd):
    """Mean frequency and spectral edge (95%)."""
    psd_sum = np.sum(psd, axis=0) + 1e-10
    mean_freq = np.mean(np.sum(freqs[:, None] * psd, axis=0) / psd_sum)
    
    cumsum = np.cumsum(psd, axis=0)
    total = cumsum[-1] + 1e-10
    sef95_vals = []
    for ch in range(psd.shape[1]):
        idx = np.searchsorted(cumsum[:, ch], 0.95 * total[ch])
        sef95_vals.append(freqs[min(idx, len(freqs)-1)])
    sef95 = np.mean(sef95_vals)
    return mean_freq, sef95

def _hjorth_params(window):
    """Hjorth activity, mobility, complexity."""
    activity = np.var(window, axis=0)
    d1 = np.diff(window, axis=0)
    var_d1 = np.var(d1, axis=0)
    mobility = np.sqrt(var_d1 / (activity + 1e-10))
    
    d2 = np.diff(d1, axis=0)
    var_d2 = np.var(d2, axis=0)
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-10)) / (mobility + 1e-10)
    return np.mean(mobility), np.mean(complexity)

def _channel_correlation(window):
    """Mean absolute inter-channel correlation."""
    corr = np.corrcoef(window.T)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.mean(np.abs(corr[mask]))

def _temporal_variability(window):
    """Coefficient of variation and line length."""
    std = np.std(window, axis=0)
    mean_abs = np.mean(np.abs(window), axis=0) + 1e-10
    cv = np.mean(std / mean_abs)
    line_len = np.mean(np.sum(np.abs(np.diff(window, axis=0)), axis=0))
    return cv, line_len

def _spectral_entropy(freqs, psd):
    """Spectral entropy (normalized)."""
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    ent = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=0)
    ent_max = np.log(psd.shape[0])
    return np.mean(ent) / ent_max

def predict(window):
    """
    Classify EEG window into source cohort probabilities.
    
    Parameters
    ----------
    window : np.ndarray
        EEG window of shape [256, 19] at 256 Hz, standardized.
    
    Returns
    -------
    probs : np.ndarray
        Probabilities [healthy_control, FTD, AD] summing to 1.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.shape != (256, 19):
        raise ValueError(f"Expected shape (256, 19), got {window.shape}")
    if not np.all(np.isfinite(window)):
        raise ValueError("Input must be finite")
    
    # Feature extraction
    bands, freqs, psd = _band_powers(window)
    mean_freq, sef95 = _spectral_features(freqs, psd)
    mobility, complexity = _hjorth_params(window)
    mean_corr = _channel_correlation(window)
    cv, line_len = _temporal_variability(window)
    spec_ent = _spectral_entropy(freqs, psd)
    
    # Derived ratios
    theta_alpha = bands['theta'] / (bands['alpha'] + 1e-10)
    delta_alpha = bands['delta'] / (bands['alpha'] + 1e-10)
    slow_total = bands['delta'] + bands['theta']
    fast_total = bands['alpha'] + bands['beta'] + bands['gamma']
    slow_fast_ratio = slow_total / (fast_total + 1e-10)
    
    # Scoring based on medical literature:
    # AD: marked slowing (high delta/theta, low alpha), reduced complexity,
    #     increased inter-channel correlation, lower spectral entropy
    # FTD: moderate slowing, variable patterns, intermediate changes
    # Healthy: alpha dominance, normal complexity, normal correlation
    
    scores = np.array([1.0, 1.0, 1.0])  # [healthy, FTD, AD]
    
    # Delta power (high = pathology, especially AD)
    if bands['delta'] > 0.35:
        scores[2] += 2.5
        scores[1] += 1.0
    elif bands['delta'] > 0.25:
        scores[2] += 1.5
        scores[1] += 0.8
    elif bands['delta'] > 0.18:
        scores[1] += 0.5
    else:
        scores[0] += 2.0
    
    # Alpha power (low = AD, normal = healthy)
    if bands['alpha'] < 0.08:
        scores[2] += 2.5
    elif bands['alpha'] < 0.12:
        scores[2] += 1.5
        scores[1] += 0.5
    elif bands['alpha'] < 0.16:
        scores[1] += 0.8
    elif bands['alpha'] > 0.20:
        scores[0] += 2.0
    
    # Theta/Alpha ratio (high = AD)
    if theta_alpha > 2.5:
        scores[2] += 2.0
    elif theta_alpha > 1.8:
        scores[2] += 1.2
        scores[1] += 0.5
    elif theta_alpha > 1.2:
        scores[1] += 0.8
    else:
        scores[0] += 1.5
    
    # Mean frequency (low = slowing)
    if mean_freq < 5.0:
        scores[2] += 1.5
    elif mean_freq < 6.5:
        scores[2] += 0.8
        scores[1] += 0.5
    elif mean_freq < 8.0:
        scores[1] += 0.5
    else:
        scores[0] += 1.2
    
    # Spectral edge frequency
    if sef95 < 12.0:
        scores[2] += 1.0
    elif sef95 < 18.0:
        scores[2] += 0.5
        scores[1] += 0.5
    elif sef95 > 22.0:
        scores[0] += 0.8
    
    # Hjorth complexity (low = AD)
    if complexity < 0.25:
        scores[2] += 1.5
    elif complexity < 0.4:
        scores[2] += 0.5
        scores[1] += 0.5
    elif complexity > 0.7:
        scores[0] += 1.0
    
    # Inter-channel correlation (high = AD - loss of differentiation)
    if mean_corr > 0.55:
        scores[2] += 1.5
    elif mean_corr > 0.40:
        scores[2] += 0.5
        scores[1] += 0.5
    elif mean_corr < 0.25:
        scores[0] += 1.0
    
    # Spectral entropy (low = AD - more regular/slow patterns)
    if spec_ent < 0.6:
        scores[2] += 1.0
    elif spec_ent < 0.75:
        scores[1] += 0.5
    else:
        scores[0] += 0.8
    
    # Temporal variability (low = AD)
    if cv < 0.4:
        scores[2] += 0.8
    elif cv > 0.8:
        scores[0] += 0.5
    
    # Line length (low = reduced complexity in AD)
    if line_len < 50:
        scores[2] += 0.5
    elif line_len > 80:
        scores[0] += 0.5
    
    # Convert to probabilities via softmax
    exp_s = np.exp(scores - np.max(scores))
    probs = exp_s / np.sum(exp_s)
    
    return probs
