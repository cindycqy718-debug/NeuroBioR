
import numpy as np
from scipy import signal

def predict(window):
    # window shape: [256, 19]
    # fs = 256 Hz
    fs = 256
    
    # 1. Compute PSD for each channel
    # Using nperseg=128 to allow for some averaging in a 256-sample window
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # 2. Average PSD across channels
    avg_psd = np.mean(psd, axis=1)
    
    # 3. Define bands
    d_mask = (freqs >= 0.5) & (freqs < 4)
    t_mask = (freqs >= 4) & (freqs < 8)
    a_mask = (freqs >= 8) & (freqs < 13)
    b_mask = (freqs >= 13) & (freqs < 30)
    
    # 4. Calculate relative power
    p_d = np.sum(avg_psd[d_mask])
    p_t = np.sum(avg_psd[t_mask])
    p_a = np.sum(avg_psd[a_mask])
    p_b = np.sum(avg_psd[b_mask])
    
    total_power = p_d + p_t + p_a + p_b + 1e-12
    rd = p_d / total_power
    rt = p_t / total_power
    ra = p_a / total_power
    rb = p_b / total_power
    
    # 5. Spectral Entropy (measure of complexity)
    # Use frequencies up to 40Hz for entropy
    se_mask = (freqs >= 0.5) & (freqs <= 40)
    psd_se = avg_psd[se_mask]
    psd_se_norm = psd_se / (np.sum(psd_se) + 1e-12)
    se = -np.sum(psd_se_norm * np.log(psd_se_norm + 1e-12)) / np.log(len(psd_se_norm))
    
    # 6. Heuristic Scores
    # Class 0: Healthy Control (HC) - Higher Alpha, Higher Beta, Higher Entropy
    # Class 1: Frontotemporal Dementia (FTD) - Higher Theta, Lower Alpha
    # Class 2: Alzheimer's Disease (AD) - Higher Delta, Lower Alpha
    
    # These coefficients are based on general EEG findings in dementia research
    score_hc = 2.5 * ra + 1.5 * rb + 1.0 * se - 1.5 * rd - 1.5 * rt
    score_ftd = 2.0 * rt + 0.5 * rd - 1.5 * ra - 1.0 * rb - 0.5 * se
    score_ad = 2.0 * rd + 1.0 * rt - 1.5 * ra - 1.0 * rb - 0.8 * se
    
    # 7. Softmax to get probabilities
    scores = np.array([score_hc, score_ftd, score_ad])
    # Numerical stability
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs
