
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def bandpower(data, sf, band, axis=0):
        band = np.asarray(band)
        low, high = band
        freqs, psd = welch(data, sf, axis=axis)
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        return np.sum(psd[idx_band], axis=0)

    sf = 256
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)

    delta_power = bandpower(window, sf, delta_band, axis=0)
    theta_power = bandpower(window, sf, theta_band, axis=0)
    alpha_power = bandpower(window, sf, alpha_band, axis=0)
    beta_power = bandpower(window, sf, beta_band, axis=0)

    total_power = delta_power + theta_power + alpha_power + beta_power
    delta_ratio = np.mean(delta_power / total_power)
    theta_ratio = np.mean(theta_power / total_power)
    alpha_ratio = np.mean(alpha_power / total_power)
    beta_ratio = np.mean(beta_power / total_power)

    channel_entropy = np.mean([entropy(np.abs(window[:, ch])) for ch in range(window.shape[1])])

    if delta_ratio > 0.4 and channel_entropy < 3.5:
        return np.array([0.8, 0.1, 0.1])
    elif theta_ratio > 0.3 and alpha_ratio < 0.2:
        return np.array([0.1, 0.7, 0.2])
    else:
        return np.array([0.1, 0.2, 0.7])
