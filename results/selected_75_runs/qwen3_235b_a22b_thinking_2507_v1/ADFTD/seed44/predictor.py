
import numpy as np
from scipy import signal

def predict(window):
    fs = 256
    freqs, psd = signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    
    delta_power = np.sum(psd[delta_mask, :], axis=0)
    theta_power = np.sum(psd[theta_mask, :], axis=0)
    alpha_power = np.sum(psd[alpha_mask, :], axis=0)
    
    avg_delta = np.mean(delta_power)
    avg_theta = np.mean(theta_power)
    avg_alpha = np.mean(alpha_power)
    
    scores = np.array([avg_alpha, avg_delta, avg_theta])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    return probs.tolist()
