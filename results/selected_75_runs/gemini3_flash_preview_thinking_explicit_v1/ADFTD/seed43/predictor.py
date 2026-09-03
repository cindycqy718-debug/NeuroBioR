
import numpy as np
import scipy.signal

def predict(window):
    # window shape: [256, 19], fs = 256 Hz
    fs = 256.0
    window = np.asarray(window)
    
    # 1. Spectral Analysis using Welch
    # nperseg=128 gives 2Hz resolution, better for a 256-sample window than 256
    freqs, psd = scipy.signal.welch(window, fs=fs, axis=0, nperseg=128)
    
    # Define bands
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    
    # Relative power per channel
    total_power = np.sum(psd, axis=0) + 1e-12
    rel_delta = np.sum(psd[delta_mask, :], axis=0) / total_power
    rel_theta = np.sum(psd[theta_mask, :], axis=0) / total_power
    rel_alpha = np.sum(psd[alpha_mask, :], axis=0) / total_power
    rel_beta = np.sum(psd[beta_mask, :], axis=0) / total_power
    
    # 2. Global Features
    m_delta = np.mean(rel_delta)
    m_theta = np.mean(rel_theta)
    m_alpha = np.mean(rel_alpha)
    m_beta = np.mean(rel_beta)
    
    # 3. Complexity: Spectral Entropy (normalized)
    psd_norm = psd / (np.sum(psd, axis=0) + 1e-12)
    se = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0) / np.log(psd.shape[0])
    m_se = np.mean(se)
    
    # 4. Spatial Features (Standard 10-20 order assumed)
    # Frontal: Fp1(0), Fp2(1), F3(2), F4(3), F7(10), F8(11), Fz(16)
    # Posterior: P3(6), P4(7), O1(8), O2(9), Pz(18)
    f_idx = [0, 1, 2, 3, 10, 11, 16]
    p_idx = [6, 7, 8, 9, 18]
    
    f_slow = np.mean(rel_theta[f_idx] + rel_delta[f_idx])
    p_slow = np.mean(rel_theta[p_idx] + rel_delta[p_idx])
    
    # 5. Heuristic Logits
    # HC: High Alpha, High Complexity, Low Slowing
    # AD: High Slowing (especially Posterior), Low Alpha, Low Complexity
    # FTD: High Slowing (especially Frontal), Low Alpha, Low Complexity
    
    # Base slowing score
    slowing = (m_theta + m_delta)
    
    # Logit for HC (Class 0)
    l0 = 1.5 * m_alpha + 2.0 * m_se - 2.0 * slowing
    
    # Logit for FTD (Class 1)
    l1 = 2.0 * slowing + 1.0 * (f_slow - p_slow) - 1.0 * m_se
    
    # Logit for AD (Class 2)
    l2 = 2.0 * slowing + 1.0 * (p_slow - f_slow) - 1.0 * m_se
    
    logits = np.array([l0, l1, l2], dtype=float)
    
    # Softmax for probabilities
    shift_logits = logits - np.max(logits)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps)
    
    return probs.tolist()
