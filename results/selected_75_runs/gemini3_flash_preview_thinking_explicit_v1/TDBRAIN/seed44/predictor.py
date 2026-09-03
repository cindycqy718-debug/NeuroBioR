import numpy as np
import scipy.signal
import scipy.stats
import math

def predict(window):
    """
    EEG source-cohort classifier for TDBRAIN dataset.
    Class 0: healthy_source_cohort
    Class 1: parkinson_source_cohort
    """
    # window shape: [256, 33], fs = 256 Hz
    fs = 256
    
    # 1. Spectral Power using Welch
    # nperseg=256 gives 1Hz resolution
    f, psd = scipy.signal.welch(window, fs=fs, axis=0, nperseg=256)
    
    # Average PSD across all 33 channels
    psd_mean = np.mean(psd, axis=1)
    
    # Define frequency bands
    delta_mask = (f >= 1) & (f < 4)
    theta_mask = (f >= 4) & (f < 8)
    alpha_mask = (f >= 8) & (f < 13)
    beta_mask = (f >= 13) & (f < 30)
    
    # Calculate band powers
    p_delta = np.sum(psd_mean[delta_mask])
    p_theta = np.sum(psd_mean[theta_mask])
    p_alpha = np.sum(psd_mean[alpha_mask])
    p_beta = np.sum(psd_mean[beta_mask])
    
    # 2. Physiological Heuristics for Parkinson's Disease (PD)
    # PD is often characterized by "slowing" of the EEG:
    # - Increased Theta and Delta power
    # - Decreased Alpha and Beta power
    
    # Avoid division by zero
    eps = 1e-10
    
    # Feature 1: Theta/Alpha ratio (Commonly increased in PD)
    tar = p_theta / (p_alpha + eps)
    
    # Feature 2: Relative Slowing (Delta + Theta) / (Alpha + Beta)
    slowing_ratio = (p_delta + p_theta) / (p_alpha + p_beta + eps)
    
    # Feature 3: Spectral Entropy (PD often shows reduced complexity)
    # Normalize PSD to get a probability distribution
    psd_norm = psd_mean / (np.sum(psd_mean) + eps)
    spec_entropy = -np.sum(psd_norm * np.log(psd_norm + eps))
    
    # 3. Combine features into a score
    # These coefficients are based on general EEG research findings for PD
    # Higher score -> higher probability of Parkinson's (Class 1)
    # We use log-transformed ratios for better distribution
    score = 0.5 * np.log(tar + 0.1) + 0.5 * np.log(slowing_ratio + 0.1) - 0.2 * spec_entropy
    
    # 4. Logistic mapping to probabilities
    # Threshold and scale are adjusted for standardized EEG windows
    # Standardized windows (mean 0, std 1) typically have total power ~1.0
    # but Welch PSD values depend on the windowing and scaling.
    # We use a bias to center the decision.
    bias = 0.5
    logit = 1.5 * (score + bias)
    
    prob_1 = 1.0 / (1.0 + math.exp(-logit))
    prob_0 = 1.0 - prob_1
    
    return np.array([float(prob_0), float(prob_1)])