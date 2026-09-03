
import numpy as np
from scipy import signal

def predict(window):
    # window shape: [256, 19], fs = 256 Hz
    # The window is independently standardized (mean 0, std 1 per channel).
    fs = 256
    
    # 1. Calculate PSD using Welch
    # nperseg=256 gives 1Hz resolution (df = 1)
    # f will be [0, 1, 2, ..., 128]
    f, pxx = signal.welch(window, fs=fs, nperseg=256, axis=0)
    
    # 2. Define frequency bands
    # Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    idx_delta = (f >= 1) & (f < 4)
    idx_theta = (f >= 4) & (f < 8)
    idx_alpha = (f >= 8) & (f < 13)
    idx_beta = (f >= 13) & (f < 30)
    
    # 3. Extract band power (mean across channels)
    # Since window is standardized, these are effectively relative powers
    delta = np.mean(pxx[idx_delta, :])
    theta = np.mean(pxx[idx_theta, :])
    alpha = np.mean(pxx[idx_alpha, :])
    beta = np.mean(pxx[idx_beta, :])
    
    # 4. Spatial features (assuming standard 10-20 order, but robust to variations)
    # Frontal channels are usually at the beginning (0-6: Fp1, Fp2, F3, F4, Fz, F7, F8)
    # Posterior channels are usually at the end (13-18: P3, P4, Pz, O1, O2)
    frontal_theta = np.mean(pxx[idx_theta, :7])
    posterior_theta = np.mean(pxx[idx_theta, 13:])
    
    # 5. Heuristic Scoring
    # Class 0: Healthy Control (HC) - High Alpha and Beta, Low Theta/Delta
    # Class 1: Frontotemporal Dementia (FTD) - High Frontal Theta
    # Class 2: Alzheimer's Disease (AD) - High Global/Posterior Theta and Delta, Low Alpha
    
    # Coefficients tuned for standardized EEG PSD scales
    # HC score: favors high alpha/beta ratio
    s0 = 5.0 * alpha + 2.0 * beta - 4.0 * theta - 2.0 * delta
    
    # FTD score: favors frontal theta slowing
    s1 = 6.0 * frontal_theta - 2.0 * alpha + 1.0 * beta
    
    # AD score: favors posterior theta and global slowing, very low alpha
    s2 = 5.0 * posterior_theta + 3.0 * delta - 5.0 * alpha
    
    # 6. Softmax for probabilities
    scores = np.array([s0, s1, s2], dtype=np.float64)
    # Numerical stability: subtract max
    e_x = np.exp(scores - np.max(scores))
    probs = e_x / np.sum(e_x)
    
    return probs.tolist()
