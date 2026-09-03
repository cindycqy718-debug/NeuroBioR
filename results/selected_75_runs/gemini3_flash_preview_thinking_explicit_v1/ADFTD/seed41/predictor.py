
import numpy as np
from scipy import signal

def predict(window):
    # window shape: [256, 19], fs = 256 Hz
    fs = 256
    
    # 1. Spectral Analysis
    # Welch PSD estimate, axis=0 is time
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # Average across channels
    psd_mean = np.mean(psd, axis=1)
    
    # Normalize PSD
    psd_sum = np.sum(psd_mean)
    if psd_sum > 0:
        psd_norm = psd_mean / psd_sum
    else:
        psd_norm = np.ones_like(psd_mean) / len(psd_mean)
    
    # Band definitions
    delta_m = (freqs >= 0.5) & (freqs < 4)
    theta_m = (freqs >= 4) & (freqs < 8)
    alpha_m = (freqs >= 8) & (freqs < 13)
    beta_m = (freqs >= 13) & (freqs < 30)
    
    delta = np.sum(psd_norm[delta_m])
    theta = np.sum(psd_norm[theta_m])
    alpha = np.sum(psd_norm[alpha_m])
    beta = np.sum(psd_norm[beta_m])
    
    # Feature 1: Slowing Ratio (AD/FTD usually show increased slow-wave activity)
    # AD often has higher slowing than FTD in posterior regions, but here we use global.
    slowing = (delta + theta) / (alpha + beta + 1e-7)
    f1 = np.log1p(slowing)
    
    # Feature 2: Spectral Entropy (Complexity)
    # HC usually has higher complexity than AD/FTD
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-7))
    f2 = entropy
    
    # Feature 3: Alpha/Theta ratio (Common marker for cognitive decline)
    at_ratio = alpha / (theta + 1e-7)
    f3 = np.log1p(at_ratio)

    # Zero-shot centroids based on literature trends:
    # Class 0 (HC): Low slowing, High entropy, High AT ratio
    # Class 1 (FTD): Medium slowing, Medium entropy, Medium AT ratio
    # Class 2 (AD): High slowing, Low entropy, Low AT ratio
    
    # Normalized feature vectors [f1, f2, f3]
    # These are heuristic estimates for standardized EEG windows
    centroids = np.array([
        [0.5, 2.5, 1.0], # HC
        [1.0, 2.0, 0.6], # FTD
        [1.5, 1.5, 0.3]  # AD
    ])
    
    current_features = np.array([f1, f2, f3])
    
    # Squared Euclidean distance
    dist = np.sum((centroids - current_features)**2, axis=1)
    
    # Softmax on negative distance
    # Using a moderate temperature to avoid over-confidence
    logits = -dist * 2.0
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return probs.tolist()
