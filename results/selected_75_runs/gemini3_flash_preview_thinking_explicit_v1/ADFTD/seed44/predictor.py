import numpy as np
import scipy.signal

def predict(window):
    """
    EEG source-cohort classifier for ADFTD dataset.
    window: np.ndarray of shape [256, 19]
    Returns: np.ndarray of shape [3] (probabilities for classes 0, 1, 2)
    """
    # Constants
    fs = 256.0
    
    # Ensure input is a numpy array
    x = np.asarray(window, dtype=np.float64)
    
    # 1. Spectral Analysis using Welch's method
    # nperseg=128 gives 2Hz resolution. fs=256.
    # axis=0 is time.
    freqs, psd = scipy.signal.welch(x, fs=fs, axis=0, nperseg=128)
    
    # Average PSD across all 19 channels
    mean_psd = np.mean(psd, axis=1)
    
    # 2. Band Power Extraction
    # Delta: 0.5-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    
    eps = 1e-12
    p_delta = np.sum(mean_psd[delta_mask])
    p_theta = np.sum(mean_psd[theta_mask])
    p_alpha = np.sum(mean_psd[alpha_mask])
    p_beta = np.sum(mean_psd[beta_mask])
    
    total_band_power = p_delta + p_theta + p_alpha + p_beta + eps
    
    # Relative Powers
    rel_delta = p_delta / total_band_power
    rel_theta = p_theta / total_band_power
    rel_alpha = p_alpha / total_band_power
    rel_beta = p_beta / total_band_power
    
    # 3. Complexity Measure: Spectral Entropy
    psd_norm = mean_psd / (np.sum(mean_psd) + eps)
    spec_entropy = -np.sum(psd_norm * np.log(psd_norm + eps))
    # Normalize entropy by log of number of frequency bins
    spec_entropy /= np.log(len(freqs))
    
    # 4. Heuristic Scoring
    # Class 0: Healthy Control - Characterized by Alpha/Beta and higher complexity.
    # Class 1: FTD - Characterized by significant Delta and Theta increases.
    # Class 2: AD - Characterized by prominent Theta increase and Alpha decrease.
    
    # Weights based on clinical literature for EEG slowing in dementia
    score_hc = 5.0 * rel_alpha + 2.0 * rel_beta + 1.0 * spec_entropy
    score_ftd = 4.5 * rel_delta + 2.5 * rel_theta - 1.0 * spec_entropy
    score_ad = 5.5 * rel_theta + 1.5 * rel_delta - 1.0 * spec_entropy
    
    # 5. Softmax Transformation
    scores = np.array([score_hc, score_ftd, score_ad], dtype=np.float64)
    # Numerical stability
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs