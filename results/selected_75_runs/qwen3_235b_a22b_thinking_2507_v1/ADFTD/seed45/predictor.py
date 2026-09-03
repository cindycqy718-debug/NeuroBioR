import numpy as np
from scipy import signal

def predict(window):
    n_channels = window.shape[1]
    total_delta = 0.0
    total_theta = 0.0
    total_alpha = 0.0
    total_beta = 0.0
    epsilon = 1e-5
    
    for ch in range(n_channels):
        f, Pxx = signal.periodogram(window[:, ch], fs=256, axis=0)
        
        delta_mask = (f >= 0.5) & (f <= 4.0)
        theta_mask = (f >= 4.0) & (f <= 8.0)
        alpha_mask = (f >= 8.0) & (f <= 12.0)
        beta_mask = (f >= 12.0) & (f <= 30.0)
        
        delta_power = np.sum(Pxx[delta_mask])
        theta_power = np.sum(Pxx[theta_mask])
        alpha_power = np.sum(Pxx[alpha_mask])
        beta_power = np.sum(Pxx[beta_mask])
        
        total_delta += delta_power
        total_theta += theta_power
        total_alpha += alpha_power
        total_beta += beta_power
    
    score0 = total_alpha / (total_theta + epsilon)
    score1 = total_beta / (total_alpha + epsilon)
    score2 = total_theta / (total_alpha + epsilon)
    
    scores = np.array([score0, score1, score2])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    return probs.tolist()