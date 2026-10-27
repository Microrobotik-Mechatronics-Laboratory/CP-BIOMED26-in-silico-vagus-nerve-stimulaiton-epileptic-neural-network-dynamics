"""
src/analysis/metrics.py
========================
Electrophysiological biomarkers: PSD, theta band power, synchrony index, and Seizure Suppression Ratio.
"""

from typing import Any
import numpy as np
from scipy.signal import welch

from config.defaults import THETA_BAND_HZ, LFP_SAMPLING_RATE_HZ


def compute_spectral_power(
    lfp: np.ndarray,
    fs: float = LFP_SAMPLING_RATE_HZ,
    band: tuple[float, float] = THETA_BAND_HZ,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Power Spectral Density (PSD) using Welch's method and integrates band power.

    Parameters
    ----------
    lfp : np.ndarray
        Filtered LFP signal.
    fs : float, optional
        Sampling frequency in Hz. Default 1024 Hz.
    band : tuple of (float, float), optional
        Frequency band of interest (low, high) in Hz. Default theta (4.0, 8.0 Hz).

    Returns
    -------
    tuple of (freqs, psd, band_power)
    """
    nperseg = min(len(lfp), int(fs * 0.5))
    if nperseg < 4:
        return np.array([0.0]), np.array([0.0]), 0.0

    nfft = max(2048, nperseg)
    freqs, psd = welch(lfp, fs=fs, nperseg=nperseg, nfft=nfft)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if np.sum(band_mask) > 1:
        band_power = float(np.trapz(psd[band_mask], freqs[band_mask]))
    elif np.sum(band_mask) == 1:
        df = freqs[1] - freqs[0] if len(freqs) > 1 else (band[1] - band[0])
        band_power = float(psd[band_mask][0] * df)
    else:
        band_power = 0.0
    return freqs, psd, band_power


def compute_firing_rate(num_spikes: int, n_neurons: int, duration_s: float) -> float:
    """Calculates mean firing rate in spikes/s/neuron."""
    if n_neurons == 0 or duration_s <= 0:
        return 0.0
    return float(num_spikes / (n_neurons * duration_s))


def compute_synchrony_index(
    spike_trains: list[np.ndarray],
    time_window_s: float,
    bin_size_ms: float = 5.0,
) -> float:
    """
    Computes population pairwise synchrony coefficient across spike trains.
    """
    if len(spike_trains) < 2:
        return 0.0

    bin_width_s = bin_size_ms * 1e-3
    n_bins = max(2, int(np.ceil(time_window_s / bin_width_s)))

    # Bin spike counts
    binned_matrix = np.zeros((len(spike_trains), n_bins), dtype=float)
    for i, spikes in enumerate(spike_trains):
        if len(spikes) > 0:
            indices = np.clip((spikes * 1e-3 / bin_width_s).astype(int), 0, n_bins - 1)
            np.add.at(binned_matrix[i], indices, 1.0)

    # Cross-correlation based synchrony (mean correlation across neuron pairs)
    std_devs = np.std(binned_matrix, axis=1)
    active_mask = std_devs > 1e-6
    active_binned = binned_matrix[active_mask]

    if active_binned.shape[0] < 2:
        return 0.0

    corr_matrix = np.corrcoef(active_binned)
    np.fill_diagonal(corr_matrix, 0.0)
    n_active = active_binned.shape[0]
    mean_corr = np.sum(corr_matrix) / (n_active * (n_active - 1))
    return float(np.clip(mean_corr, 0.0, 1.0))


def compute_seizure_suppression_ratio(
    theta_power_vns: float,
    theta_power_baseline: float,
) -> float:
    """
    Computes Seizure Suppression Ratio (SSR):
    SSR = 1 - (P_theta,VNS / P_theta,Baseline).
    SSR > 0 indicates suppression of hypersynchronous ictal power.
    SSR = 1.0 represents complete suppression.
    """
    if theta_power_baseline <= 1e-25:
        return 0.0
    ssr = 1.0 - (theta_power_vns / theta_power_baseline)
    return float(np.clip(ssr, -1.0, 1.0))
