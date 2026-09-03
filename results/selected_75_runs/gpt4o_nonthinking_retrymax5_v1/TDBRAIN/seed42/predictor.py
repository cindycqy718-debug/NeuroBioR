
import numpy as np
import scipy.signal
import scipy.stats

def predict(window):
    def bandpower(data, sf, band, axis=0):
        band = np.asarray(band)
        low, high = band
        freqs, psd = scipy.signal.welch(data, sf, axis=axis)
        band_power = np.sum(psd[(freqs >= low) & (freqs <= high)], axis=0)
        return band_power

    def spectral_entropy(data, sf, band, axis=0):
        band = np.asarray(band)
        low, high = band
        freqs, psd = scipy.signal.welch(data, sf, axis=axis)
        psd_band = psd[(freqs >= low) & (freqs <= high)]
        psd_norm = psd_band / np.sum(psd_band, axis=0, keepdims=True)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=0)
        return entropy

    sf = 256  # Sampling frequency
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    delta_power = bandpower(window, sf, delta_band, axis=0)
    theta_power = bandpower(window, sf, theta_band, axis=0)
    alpha_power = bandpower(window, sf, alpha_band, axis=0)
    beta_power = bandpower(window, sf, beta_band, axis=0)

    total_power = delta_power + theta_power + alpha_power + beta_power
    relative_alpha_power = alpha_power / (total_power + 1e-10)
    relative_beta_power = beta_power / (total_power + 1e-10)

    alpha_entropy = spectral_entropy(window, sf, alpha_band, axis=0)
    beta_entropy = spectral_entropy(window, sf, beta_band, axis=0)

    mean_relative_alpha = np.mean(relative_alpha_power)
    mean_relative_beta = np.mean(relative_beta_power)
    mean_alpha_entropy = np.mean(alpha_entropy)
    mean_beta_entropy = np.mean(beta_entropy)

    # Decision rule based on features
    if mean_relative_alpha > mean_relative_beta and mean_alpha_entropy < mean_beta_entropy:
        return np.array([0.9, 0.1])  # Likely healthy_source_cohort
    else:
        return np.array([0.1, 0.9])  # Likely parkinson_source_cohort
