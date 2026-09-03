import numpy as np
from scipy.signal import welch

def predict(window):
    fs = 256
    f, Pxx = welch(window, fs=fs, nperseg=128, noverlap=64, axis=0)
    
    # Frequency band definitions (exclusive upper bounds)
    delta_idx = np.where((f >= 0.5) & (f < 4))[0]
    theta_idx = np.where((f >= 4) & (f < 8))[0]
    alpha_idx = np.where((f >= 8) & (f < 12))[0]
    
    # Band power integration and channel averaging
    delta_power = np.mean(np.sum(Pxx[delta_idx], axis=0)) if len(delta_idx) else 0.0
    theta_power = np.mean(np.sum(Pxx[theta_idx], axis=0)) if len(theta_idx) else 0.0
    alpha_power = np.mean(np.sum(Pxx[alpha_idx], axis=0)) if len(alpha_idx) else 0.0
    
    # Class scores based on band power composition
    scores = np.array([alpha_power, delta_power, theta_power])
    
    # Convert to probabilities via softmax
    exp_scores = np.exp(scores - np.max(scores))
    return (exp_scores / exp_scores.sum()).tolist()
