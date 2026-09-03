import numpy as np
from scipy import signal, stats
import math

def _band_power(psd, freqs, fmin, fmax):
    """Compute relative band power in frequency range."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.mean(psd[:, mask]))

def _hjorth_params(x, axis=0):
    """Compute Hjorth activity, mobility, complexity."""
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    
    var_x = np.var(x, axis=axis)
    var_dx = np.var(dx, axis=axis)
    var_ddx = np.var(ddx, axis=axis)
    
    var_x = np.maximum(var_x, 1e-12)
    var_dx = np.maximum(var_dx, 1e-12)
    
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility
    
    return np.mean(activity), np.mean(mobility), np.mean(complexity)

def _spectral_entropy(psd):
    """Compute spectral entropy normalized to [0,1]."""
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    psd_norm = np.clip(psd_norm, 1e-12, 1.0)
    ent = -np.sum(psd_norm * np.log(psd_norm), axis=1)
    max_ent = np.log(psd.shape[1])
    return float(np.mean(ent) / max_ent) if max_ent > 0 else 0.5

def _signal_entropy(x, axis=0):
    """Compute approximate signal entropy via histogram."""
    ent_sum = 0.0
    n_chan = x.shape[1] if axis == 0 else x.shape[0]
    for c in range(n_chan):
        col = x[:, c] if axis == 0 else x[c, :]
        hist, _ = np.histogram(col, bins=10, density=True)
        hist = hist[hist > 0]
        ent_sum += -np.sum(hist * np.log(hist + 1e-12))
    return ent_sum / n_chan

def _channel_correlation(x):
    """Compute mean absolute correlation across channels."""
    if x.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(x.T)
    off_diag = corr[np.triu_indices(x.shape[1], k=1)]
    return float(np.mean(np.abs(off_diag)))

def predict(window):
    """
    Deterministic EEG source-cohort classifier.
    
    Input: window [256, 33] at 256 Hz
    Output: [p_healthy, p_parkinson] probabilities summing to 1
    
    Physiological basis:
    - PD EEG shows spectral slowing (increased delta/theta, decreased alpha)
    - Altered spectral entropy and complexity
    - Changes in cross-channel synchronization
    """
    # Validate input
    window = np.asarray(window, dtype=np.float64)
    assert window.shape == (256, 33), f"Expected shape (256, 33), got {window.shape}"
    assert np.all(np.isfinite(window)), "Input must be finite"
    
    fs = 256.0
    nperseg = min(128, window.shape[0])
    
    # Compute PSD using Welch method (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=nperseg)
    psd = psd.T  # [channels, freqs]
    
    # Band definitions (Hz)
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    # Compute relative band powers
    total_power = np.sum(psd, axis=1, keepdims=True) + 1e-12
    rel_powers = {}
    for name, (fmin, fmax) in bands.items():
        bp = _band_power(psd / total_power, freqs, fmin, fmax)
        rel_powers[name] = bp
    
    # Spectral features
    spectral_ent = _spectral_entropy(psd)
    
    # Spectral edge frequency (95% power)
    cumsum = np.cumsum(psd, axis=1)
    total = cumsum[:, -1:] + 1e-12
    threshold = 0.95 * total
    sef95 = []
    for ch in range(psd.shape[0]):
        idx = np.searchsorted(cumsum[ch, :], threshold[ch, 0])
        sef95.append(freqs[min(idx, len(freqs)-1)])
    sef95_mean = float(np.mean(sef95))
    
    # Hjorth parameters
    activity, mobility, complexity = _hjorth_params(window, axis=0)
    
    # Temporal variability
    var_mean = float(np.mean(np.var(window, axis=0)))
    cv_mean = float(np.mean(np.std(window, axis=0) / (np.mean(np.abs(window), axis=0) + 1e-12)))
    
    # Signal entropy
    sig_ent = _signal_entropy(window, axis=0)
    
    # Cross-channel correlation
    corr_mean = _channel_correlation(window)
    
    # === PD Scoring based on physiological literature ===
    # PD typically shows:
    # - Increased delta/theta (slowing)
    # - Decreased alpha
    # - Increased spectral entropy (more disorganized)
    # - Altered complexity
    # - Changes in connectivity
    
    score = 0.0
    
    # Spectral slowing: high delta+theta, low alpha suggests PD
    slow_power = rel_powers['delta'] + rel_powers['theta']
    if slow_power > 0.35:
        score += 1.5
    elif slow_power > 0.25:
        score += 0.8
    elif slow_power > 0.18:
        score += 0.3
    
    # Alpha suppression in PD
    alpha = rel_powers['alpha']
    if alpha < 0.15:
        score += 1.0
    elif alpha < 0.22:
        score += 0.5
    elif alpha > 0.35:
        score -= 0.5  # High alpha suggests healthy
    
    # Beta power changes
    beta = rel_powers['beta']
    if beta > 0.25:
        score += 0.5  # Elevated beta in PD
    elif beta < 0.12:
        score -= 0.3
    
    # Spectral entropy: higher in PD (more disorganized)
    if spectral_ent > 0.85:
        score += 1.0
    elif spectral_ent > 0.75:
        score += 0.5
    elif spectral_ent < 0.55:
        score -= 0.5
    
    # Spectral edge frequency: lower in PD (slowing)
    if sef95_mean < 15.0:
        score += 0.8
    elif sef95_mean < 20.0:
        score += 0.4
    elif sef95_mean > 30.0:
        score -= 0.4
    
    # Hjorth complexity: altered in PD
    if complexity > 1.5:
        score += 0.5
    elif complexity < 0.8:
        score -= 0.3
    
    # Mobility: related to mean frequency
    if mobility < 0.3:
        score += 0.4  # Lower mobility (slower signals) in PD
    
    # Signal entropy
    if sig_ent > 2.0:
        score += 0.5
    
    # Cross-channel correlation: altered connectivity in PD
    if corr_mean > 0.4:
        score += 0.3  # Hyper-connectivity
    elif corr_mean < 0.15:
        score += 0.2  # Or hypo-connectivity
    
    # Convert score to probability using sigmoid
    # Center the sigmoid so score=0 gives ~50% probability
    # Positive score -> more PD-like
    score_normalized = score / 5.0  # Scale factor
    p_pd = 1.0 / (1.0 + math.exp(-score_normalized))
    
    # Ensure valid probability range
    p_pd = float(np.clip(p_pd, 0.01, 0.99))
    p_healthy = 1.0 - p_pd
    
    return np.array([p_healthy, p_pd], dtype=np.float64)
