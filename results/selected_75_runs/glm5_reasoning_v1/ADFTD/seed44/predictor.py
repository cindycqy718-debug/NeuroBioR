import numpy as np
from scipy import signal
from scipy.stats import entropy as scipy_entropy
import math

def predict(window):
    """
    Deterministic zero-shot EEG classifier for AD/FTD/healthy classification.
    Input: window [256, 19] at 256 Hz
    Output: probabilities [3] for [healthy, FTD, AD]
    """
    n_samples, n_channels = window.shape
    fs = 256.0
    
    # === BAND POWER EXTRACTION ===
    # Welch PSD with explicit axis=0 (time axis)
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=min(64, n_samples))
    
    # Standard EEG frequency bands
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    band_powers = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            band_powers[name] = np.mean(psd[mask, :], axis=0)
        else:
            band_powers[name] = np.zeros(n_channels)
    
    # Global band powers (mean across channels)
    delta_power = np.mean(band_powers['delta'])
    theta_power = np.mean(band_powers['theta'])
    alpha_power = np.mean(band_powers['alpha'])
    beta_power = np.mean(band_powers['beta'])
    
    # === KEY RATIOS (AD biomarkers) ===
    theta_alpha_ratio = theta_power / (alpha_power + 1e-10)
    delta_alpha_ratio = delta_power / (alpha_power + 1e-10)
    delta_theta_ratio = delta_power / (theta_power + 1e-10)
    
    # === ALPHA PEAK FREQUENCY (slowing in AD) ===
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    if np.any(alpha_mask):
        alpha_psd = psd[alpha_mask, :]
        alpha_freqs = freqs[alpha_mask]
        peak_idx = np.argmax(np.mean(alpha_psd, axis=1))
        alpha_peak_freq = alpha_freqs[peak_idx]
    else:
        alpha_peak_freq = 10.0
    
    # === FRONTAL-POSTERIOR ASYMMETRY (FTD marker) ===
    # Standard 10-20 channel ordering assumption: frontal in first channels
    frontal_channels = band_powers['delta'][:6]  # Fp1, Fp2, F3, F4, F7, F8 region
    posterior_channels = band_powers['delta'][14:19]  # P3, P4, O1, O2 region
    
    frontal_delta = np.mean(frontal_channels)
    posterior_delta = np.mean(posterior_channels)
    fp_delta_ratio = frontal_delta / (posterior_delta + 1e-10)
    
    frontal_alpha = np.mean(band_powers['alpha'][:6])
    posterior_alpha = np.mean(band_powers['alpha'][14:19])
    fp_alpha_ratio = frontal_alpha / (posterior_alpha + 1e-10)
    
    # === SPECTRAL ENTROPY (complexity measure) ===
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    psd_flat = psd_norm.flatten()
    psd_flat = psd_flat / (np.sum(psd_flat) + 1e-10)
    spectral_entropy = scipy_entropy(psd_flat + 1e-10)
    
    # === TEMPORAL VARIABILITY ===
    channel_stds = np.std(window, axis=0)
    channel_means = np.mean(np.abs(window), axis=0)
    cv = np.mean(channel_stds / (channel_means + 1e-10))
    
    # === CROSS-CHANNEL CORRELATION (functional connectivity) ===
    corr_matrix = np.corrcoef(window.T)
    # Mean absolute correlation (excluding diagonal)
    mask = ~np.eye(n_channels, dtype=bool)
    mean_corr = np.mean(np.abs(corr_matrix[mask]))
    
    # === HJORTH PARAMETERS ===
    # Activity (variance)
    activity = np.mean(np.var(window, axis=0))
    
    # Mobility (mean frequency)
    diff1 = np.diff(window, axis=0)
    mobility = np.mean(np.std(diff1, axis=0) / (channel_stds + 1e-10))
    
    # Complexity
    diff2 = np.diff(diff1, axis=0)
    complexity = np.mean(np.std(diff2, axis=0) / (np.std(diff1, axis=0) + 1e-10))
    
    # === CLASSIFICATION RULES ===
    # Based on established EEG literature:
    # - AD: increased slow waves, decreased alpha, posterior slowing
    # - FTD: frontal changes, different pattern from AD
    # - Healthy: preserved alpha, low slow/alpha ratios
    
    # Normalize features to [0, 1] range using sigmoid-like transforms
    tar_score = 1.0 / (1.0 + math.exp(-5.0 * (theta_alpha_ratio - 0.8)))
    dar_score = 1.0 / (1.0 + math.exp(-5.0 * (delta_alpha_ratio - 0.5)))
    apf_score = 1.0 / (1.0 + math.exp(2.0 * (alpha_peak_freq - 9.5)))  # Low APF -> high score
    
    # Frontal dominance score (higher in FTD)
    frontal_score = 1.0 / (1.0 + math.exp(-3.0 * (fp_delta_ratio - 1.0)))
    
    # Entropy score
    entropy_score = spectral_entropy / 5.0  # Normalize roughly
    
    # === COMPUTE CLASS SCORES ===
    # Healthy: low slow/alpha ratios, high alpha peak, low entropy
    healthy_score = (1.0 - tar_score) * 0.4 +                     (1.0 - dar_score) * 0.3 +                     (1.0 - apf_score) * 0.15 +                     (1.0 - entropy_score) * 0.15
    
    # FTD: moderate slow ratios, high frontal involvement, high entropy
    ftd_score = tar_score * 0.25 +                 frontal_score * 0.35 +                 entropy_score * 0.25 +                 (1.0 - apf_score) * 0.15
    
    # AD: high slow/alpha ratios, low alpha peak, posterior slowing
    ad_score = tar_score * 0.35 +                dar_score * 0.30 +                apf_score * 0.20 +                (1.0 - frontal_score) * 0.15
    
    # === CONVERT TO PROBABILITIES ===
    scores = np.array([healthy_score, ftd_score, ad_score])
    
    # Softmax transformation
    scores_shifted = scores - np.max(scores)  # Numerical stability
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs
