import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx], axis=0)

def spectral_entropy(psd, freqs):
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    return np.sum(-psd_norm * np.log(psd_norm + 1e-12), axis=0)

def predict(window):
    fs = 256
    freqs, psd = welch(window, fs=fs, axis=0, nperseg=128, noverlap=64)
    delta = (0.5, 4)
    alpha = (8, 13)
    delta_power = bandpower(psd, freqs, delta)
    alpha_power = bandpower(psd, freqs, alpha)
    total_power = np.mean(psd, axis=0)
    rel_alpha = alpha_power / (total_power + 1e-12)
    spec_ent = spectral_entropy(psd, freqs)
    mean_spec_ent = np.mean(spec_ent)
    temporal_var = np.mean(np.std(window, axis=0))
    corr = np.corrcoef(window.T)
    upper_tri = corr[np.triu_indices_from(corr, k=1)]
    mean_corr = np.mean(upper_tri)
    delta_norm = np.clip((delta_power - 0.5) / 2.0, 0, 1)
    alpha_norm = np.clip((rel_alpha - 0.1) / 0.3, 0, 1)
    spec_ent_norm = np.clip((mean_spec_ent - 1.5) / 0.5, 0, 1)
    temp_var_norm = np.clip((temporal_var - 0.5) / 1.0, 0, 1)
    corr_norm = np.clip((mean_corr - 0.3) / 0.4, 0, 1)
    alz_score = np.sum(delta_norm) + np.sum(1 - alpha_norm) + 5*(1 - spec_ent_norm) + 5*(1 - temp_var_norm) + 5*(1 - corr_norm)
    healthy_score = 5*16 - alz_score
    max_score = max(healthy_score, alz_score)
    exp_healthy = np.exp(healthy_score - max_score)
    exp_alz = np.exp(alz_score - max_score)
    prob_healthy = exp_healthy / (exp_healthy + exp_alz)
    prob_alz = exp_alz / (exp_healthy + exp_alz)
    return np.array([prob_healthy, prob_alz])
