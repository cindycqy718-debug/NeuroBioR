
import numpy as np
import scipy.signal
import scipy.stats

def predict(window):
    # window shape [256, 33], fs = 256 Hz
    # Axis 0 is time, Axis 1 is channels
    fs = 256.0
    
    # 1. Compute Power Spectral Density (PSD)
    # nperseg=256 gives 1Hz frequency resolution
    freqs, psd = scipy.signal.welch(window, fs=fs, nperseg=256, axis=0)
    
    # 2. Define frequency bands
    # Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    delta_mask = (freqs >= 1) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    # 3. Calculate band power (average across channels)
    # psd shape: [freq_bins, 33]
    avg_psd = np.mean(psd, axis=1)
    
    p_delta = np.sum(avg_psd[delta_mask])
    p_theta = np.sum(avg_psd[theta_mask])
    p_alpha = np.sum(avg_psd[alpha_mask])
    p_beta = np.sum(avg_psd[beta_mask])
    
    # 4. Physiological Heuristic: Parkinson's often shows EEG slowing
    # Increased Theta/Alpha ratio or (Delta+Theta)/(Alpha+Beta)
    # We use a combined slowing index
    total_power = p_delta + p_theta + p_alpha + p_beta + 1e-9
    slowing_index = (p_delta + p_theta) / (p_alpha + p_beta + 1e-9)
    
    # 5. Complexity measure: Spectral Entropy
    psd_norm = avg_psd / (np.sum(avg_psd) + 1e-9)
    spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-9))
    
    # 6. Decision Logic (Zero-shot heuristic)
    # Parkinson's cohort (class 1) typically shows higher slowing and lower complexity
    # These thresholds are based on general EEG literature for PD vs Healthy
    # Standardized windows mean power levels are relative.
    
    # Heuristic score: higher means more likely Parkinson's
    # Slowing index > 1.0 is a common threshold for "abnormal" slowing
    # Spectral entropy for healthy is usually higher.
    score = (slowing_index - 0.8) * 2.0 - (spec_entropy - 3.0) * 0.5
    
    # Sigmoid to get probability for class 1
    prob1 = 1.0 / (1.0 + np.exp(-score))
    prob0 = 1.0 - prob1
    
    return np.array([prob0, prob1], dtype=float)
