import numpy as np
from scipy import signal

def predict(window):
    f, Pxx = signal.welch(window, fs=256, nperseg=256, axis=0)
    delta_mask = (f >= 1) & (f <= 4)
    theta_mask = (f >= 4) & (f <= 8)
    alpha_mask = (f >= 8) & (f <= 12)
    beta_mask = (f >= 13) & (f <= 30)
    delta_power = np.mean(np.sum(Pxx[delta_mask, :], axis=0))
    theta_power = np.mean(np.sum(Pxx[theta_mask, :], axis=0))
    alpha_power = np.mean(np.sum(Pxx[alpha_mask, :], axis=0))
    beta_power = np.mean(np.sum(Pxx[beta_mask, :], axis=0))
    score_0 = alpha_power
    score_1 = beta_power
    score_2 = delta_power
    scores = np.array([score_0, score_1, score_2])
    total = np.sum(scores)
    if total == 0:
        return [1/3, 1/3, 1/3]
    return (scores / total).tolist()
