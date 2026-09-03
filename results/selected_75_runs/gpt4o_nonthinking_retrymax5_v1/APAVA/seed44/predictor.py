
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def compute_band_powers(window, fs=256):
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    band_powers = {}
    for band, (low, high) in bands.items():
        f, Pxx = welch(window, fs=fs, axis=0, nperseg=fs)
        band_power = np.sum(Pxx[(f >= low) & (f < high)], axis=0)
        band_powers[band] = np.mean(band_power)
    return band_powers

def compute_entropy(window):
    return np.mean([entropy(np.abs(channel)) for channel in window.T])

def predict(window):
    if window.shape != (256, 16):
        raise ValueError("Input window must have shape [256, 16]")
    
    band_powers = compute_band_powers(window)
    entropy_value = compute_entropy(window)
    
    # Rule-based classification
    if band_powers['theta'] > band_powers['alpha'] and entropy_value > 3.5:
        return np.array([0.2, 0.8])  # Higher probability for Alzheimer's source cohort
    else:
        return np.array([0.8, 0.2])  # Higher probability for Healthy source cohort
