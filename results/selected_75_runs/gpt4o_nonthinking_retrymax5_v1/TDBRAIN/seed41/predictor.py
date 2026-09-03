
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def predict(window):
    def bandpower(data, sf, band, axis=0):
        band = np.asarray(band)
        low, high = band
        freqs, psd = welch(data, sf, axis=axis)
        band_power = np.sum(psd[(freqs >= low) & (freqs <= high)], axis=0)
        return band_power

    def spectral_entropy(data, sf, axis=0):
        freqs, psd = welch(data, sf, axis=axis)
        psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
        return entropy(psd_norm, axis=0)

    sf = 256  # Sampling frequency
    delta_power = bandpower(window, sf, [0.5, 4], axis=0)
    beta_power = bandpower(window, sf, [13, 30], axis=0)
    entropy_vals = spectral_entropy(window, sf, axis=0)

    delta_mean = np.mean(delta_power)
    beta_mean = np.mean(beta_power)
    entropy_mean = np.mean(entropy_vals)

    # Fixed rule-based classification
    if delta_mean > beta_mean and entropy_mean < 2.5:
        return np.array([0.9, 0.1])  # Likely healthy_source_cohort
    else:
        return np.array([0.1, 0.9])  # Likely parkinson_source_cohort
