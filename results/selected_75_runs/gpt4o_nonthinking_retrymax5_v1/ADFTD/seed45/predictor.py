
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    if window.shape != (256, 19):
        raise ValueError("Input window must have shape [256, 19].")
    if not np.isfinite(window).all():
        raise ValueError("Input window contains non-finite values.")

    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    fs = 256

    freqs, psd = welch(window, fs=fs, axis=0, nperseg=128)

    delta_power = np.zeros(window.shape[1])
    theta_power = np.zeros(window.shape[1])
    alpha_power = np.zeros(window.shape[1])
    beta_power = np.zeros(window.shape[1])
    total_power = np.zeros(window.shape[1])

    for i in range(window.shape[1]):
        delta_power[i] = np.sum(psd[(freqs >= delta_band[0]) & (freqs < delta_band[1]), i])
        theta_power[i] = np.sum(psd[(freqs >= theta_band[0]) & (freqs < theta_band[1]), i])
        alpha_power[i] = np.sum(psd[(freqs >= alpha_band[0]) & (freqs < alpha_band[1]), i])
        beta_power[i] = np.sum(psd[(freqs >= beta_band[0]) & (freqs < beta_band[1]), i])
        total_power[i] = np.sum(psd[:, i])

    total_power[total_power == 0] = np.finfo(float).eps

    delta_ratio = np.mean(delta_power / total_power)
    theta_ratio = np.mean(theta_power / total_power)
    alpha_ratio = np.mean(alpha_power / total_power)
    beta_ratio = np.mean(beta_power / total_power)

    temporal_variability = np.mean(np.std(window, axis=0))
    spectral_entropy = np.mean([entropy(psd[:, i] / np.sum(psd[:, i])) for i in range(window.shape[1])])

    if delta_ratio > 0.4 and spectral_entropy < 2.5:
        probabilities = np.array([0.1, 0.8, 0.1])
    elif alpha_ratio > 0.3 and beta_ratio > 0.2:
        probabilities = np.array([0.1, 0.1, 0.8])
    else:
        probabilities = np.array([0.8, 0.1, 0.1])

    probabilities /= np.sum(probabilities)
    return probabilities
