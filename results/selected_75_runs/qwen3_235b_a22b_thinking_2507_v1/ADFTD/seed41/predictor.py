import numpy as np
from scipy import signal

def predict(window):
    fs = 256
    f, Pxx = signal.welch(window, fs=fs, nperseg=256, axis=0)
    
    delta_idx = np.where((f >= 1) & (f < 4))[0]
    theta_idx = np.where((f >= 4) & (f < 8))[0]
    alpha_idx = np.where((f >= 8) & (f < 13))[0]
    beta_idx = np.where((f >= 13) & (f < 30))[0]
    gamma_idx = np.where((f >= 30) & (f <= 100))[0]
    
    def band_power(idx):
        return np.sum(Pxx[idx, :]) if len(idx) > 0 else 0.0
    
    delta = band_power(delta_idx)
    theta = band_power(theta_idx)
    alpha = band_power(alpha_idx)
    beta = band_power(beta_idx)
    gamma = band_power(gamma_idx)
    
    total = delta + theta + alpha + beta + gamma
    if total == 0:
        return [1/3, 1/3, 1/3]
    
    rel_alpha = alpha / total
    rel_beta = beta / total
    rel_slow = (delta + theta) / total
    
    scores = np.array([rel_alpha, rel_beta, rel_slow])
    exp_scores = np.exp(scores - np.max(scores))
    return (exp_scores / exp_scores.sum()).tolist()