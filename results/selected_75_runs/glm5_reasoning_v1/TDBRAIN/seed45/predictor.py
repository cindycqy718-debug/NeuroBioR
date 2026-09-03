import numpy as np
from scipy import signal, stats
import math

def compute_band_powers(psd, freqs, fs):
    """Compute relative band powers from PSD."""
    # Define frequency bands
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'lowgamma': (30.0, 45.0)
    }
    
    band_powers = {}
    total_power = np.zeros(psd.shape[1])
    
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.sum(mask) > 0:
            band_power = np.mean(psd[mask, :], axis=0)
            band_powers[band_name] = band_power
            total_power += band_power
        else:
            band_powers[band_name] = np.zeros(psd.shape[1])
    
    # Relative powers
    rel_powers = {}
    for band_name in bands:
        rel_powers[band_name] = np.where(total_power > 1e-12, 
                                          band_powers[band_name] / total_power, 
                                          0.0)
    
    return rel_powers

def compute_hjorth_params(window):
    """Compute Hjorth parameters for each channel."""
    # Activity: variance of the signal
    activity = np.var(window, axis=0)
    
    # Mobility: sqrt(variance of first derivative / variance of signal)
    first_diff = np.diff(window, axis=0)
    mobility = np.sqrt(np.var(first_diff, axis=0) / (activity + 1e-12))
    
    # Complexity: mobility of first derivative / mobility of signal
    second_diff = np.diff(first_diff, axis=0)
    mobility_deriv = np.sqrt(np.var(second_diff, axis=0) / (np.var(first_diff, axis=0) + 1e-12))
    complexity = mobility_deriv / (mobility + 1e-12)
    
    return activity, mobility, complexity

def compute_spectral_entropy(psd):
    """Compute spectral entropy for each channel."""
    # Normalize PSD to probability distribution
    psd_sum = np.sum(psd, axis=0, keepdims=True)
    psd_norm = psd / (psd_sum + 1e-12)
    
    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=0)
    
    # Normalize by max entropy (log2(N))
    max_entropy = np.log2(psd.shape[0])
    return entropy / max_entropy

def compute_cross_channel_corr(window):
    """Compute mean absolute cross-channel correlation."""
    n_channels = window.shape[1]
    corr_matrix = np.corrcoef(window.T)
    
    # Get upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
    
    return np.mean(np.abs(upper_tri)), np.std(np.abs(upper_tri))

def predict(window):
    """
    Predict source cohort from EEG window.
    
    Parameters:
    -----------
    window : np.ndarray
        EEG data shaped [256, 33] at 256 Hz
        
    Returns:
    --------
    np.ndarray
        Probabilities [p_healthy, p_parkinson] summing to 1
    """
    # Validate input
    window = np.asarray(window, dtype=np.float64)
    assert window.shape == (256, 33), f"Expected shape (256, 33), got {window.shape}"
    assert np.all(np.isfinite(window)), "Input contains non-finite values"
    
    fs = 256.0
    n_channels = 33
    
    # Compute PSD using Welch method (axis=0 for time axis)
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(64, window.shape[0]), axis=0)
    
    # 1. Band power features
    rel_powers = compute_band_powers(psd, freqs, fs)
    
    # Mean relative powers across channels
    delta_rel = np.mean(rel_powers['delta'])
    theta_rel = np.mean(rel_powers['theta'])
    alpha_rel = np.mean(rel_powers['alpha'])
    beta_rel = np.mean(rel_powers['beta'])
    lowgamma_rel = np.mean(rel_powers['lowgamma'])
    
    # Delta/alpha ratio (elevated in Parkinson's - slowing)
    delta_alpha_ratio = delta_rel / (alpha_rel + 1e-12)
    
    # Theta/alpha ratio
    theta_alpha_ratio = theta_rel / (alpha_rel + 1e-12)
    
    # 2. Hjorth parameters
    activity, mobility, complexity = compute_hjorth_params(window)
    mean_mobility = np.mean(mobility)
    mean_complexity = np.mean(complexity)
    
    # 3. Spectral entropy (reduced in Parkinson's - less complex)
    spectral_entropy = compute_spectral_entropy(psd)
    mean_spectral_entropy = np.mean(spectral_entropy)
    
    # 4. Temporal variability
    signal_std = np.std(window, axis=0)
    mean_std = np.mean(signal_std)
    
    # Coefficient of variation across channels
    cv_std = np.std(signal_std) / (mean_std + 1e-12)
    
    # 5. Cross-channel correlation
    mean_corr, std_corr = compute_cross_channel_corr(window)
    
    # 6. Spectral edge frequency (95% power)
    cumsum_psd = np.cumsum(psd, axis=0)
    total_power = cumsum_psd[-1, :]
    sef95_idx = np.argmax(cumsum_psd >= 0.95 * total_power[np.newaxis, :], axis=0)
    sef95 = freqs[sef95_idx]
    mean_sef95 = np.mean(sef95)
    
    # --- Decision rule based on Parkinson's EEG characteristics ---
    # Parkinson's typically shows:
    # - Increased slow wave activity (delta, theta)
    # - Decreased alpha
    # - Reduced spectral entropy (less complex)
    # - Altered connectivity
    
    # Score accumulation for Parkinson's likelihood
    score = 0.0
    
    # 1. Delta/alpha ratio (higher in PD - brain slowing)
    if delta_alpha_ratio > 0.8:
        score += 1.5
    elif delta_alpha_ratio > 0.5:
        score += 0.8
    elif delta_alpha_ratio < 0.2:
        score -= 0.5
    
    # 2. Theta/alpha ratio (higher in PD)
    if theta_alpha_ratio > 0.6:
        score += 1.0
    elif theta_alpha_ratio > 0.4:
        score += 0.5
    elif theta_alpha_ratio < 0.15:
        score -= 0.3
    
    # 3. Alpha power (reduced in PD)
    if alpha_rel < 0.15:
        score += 1.2
    elif alpha_rel < 0.25:
        score += 0.6
    elif alpha_rel > 0.35:
        score -= 0.5
    
    # 4. Beta power (elevated in PD motor cortex)
    if beta_rel > 0.35:
        score += 0.8
    elif beta_rel > 0.25:
        score += 0.3
    
    # 5. Spectral entropy (reduced in PD - less complex dynamics)
    if mean_spectral_entropy < 0.6:
        score += 1.0
    elif mean_spectral_entropy < 0.75:
        score += 0.4
    elif mean_spectral_entropy > 0.85:
        score -= 0.4
    
    # 6. Hjorth complexity (reduced in PD)
    if mean_complexity < 0.5:
        score += 0.8
    elif mean_complexity < 0.8:
        score += 0.3
    elif mean_complexity > 1.2:
        score -= 0.3
    
    # 7. Spectral edge frequency (lower in PD - slowing)
    if mean_sef95 < 15.0:
        score += 0.8
    elif mean_sef95 < 20.0:
        score += 0.3
    elif mean_sef95 > 30.0:
        score -= 0.3
    
    # 8. Cross-channel correlation (altered in PD)
    if mean_corr > 0.6:
        score += 0.5  # Hyperconnectivity in some PD
    elif mean_corr < 0.2:
        score += 0.3  # Or hypoconnectivity
    
    # 9. Mobility (reduced in PD - less variable)
    if mean_mobility < 0.15:
        score += 0.5
    elif mean_mobility > 0.3:
        score -= 0.3
    
    # Convert score to probability using sigmoid-like transformation
    # Score range approximately -2 to +8, center around 0
    # Map to probability using logistic function
    
    # Normalize score to reasonable range
    normalized_score = (score - 2.0) / 3.0  # Center around 2, scale by 3
    
    # Logistic function
    p_parkinson = 1.0 / (1.0 + np.exp(-normalized_score))
    
    # Ensure valid probability range
    p_parkinson = np.clip(p_parkinson, 0.01, 0.99)
    p_healthy = 1.0 - p_parkinson
    
    return np.array([p_healthy, p_parkinson], dtype=np.float64)
