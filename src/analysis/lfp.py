"""
src/analysis/lfp.py
====================
Local Field Potential (LFP) simulation and filtering.
Based on transmembrane synaptic currents and dipolar summation (Aussel et al. 2022, Mazzoni et al. 2015).
"""

from typing import Any, Optional
import numpy as np
from scipy.signal import butter, filtfilt, resample

from config.defaults import (
    LFP_SAMPLING_RATE_HZ,
    LFP_FILTER_LOW_HZ,
    LFP_FILTER_HIGH_HZ,
    LIF_E_SYN_EX_MV,
    LIF_E_SYN_IN_NORMAL_MV,
    LIF_E_SYN_IN_DEP_MV,
    CHLORIDE_THRESHOLD,
)


def compute_lfp_from_states(
    v_arr: np.ndarray,
    ge_arr: np.ndarray,
    gi_arr: np.ndarray,
    time_ms: np.ndarray,
    cl_arr: Optional[np.ndarray] = None,
    dipole_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes population LFP from membrane potentials and synaptic conductances.

    Parameters
    ----------
    v_arr : np.ndarray
        Membrane potentials (mV), shape (N_neurons, N_timepoints).
    ge_arr : np.ndarray
        Excitatory synaptic conductances (siemens), shape (N_neurons, N_timepoints).
    gi_arr : np.ndarray
        Inhibitory synaptic conductances (siemens), shape (N_neurons, N_timepoints).
    time_ms : np.ndarray
        Simulation time vector in milliseconds.
    cl_arr : np.ndarray, optional
        Intracellular chloride concentrations. If None, assumes normal E_rev = -80 mV.
    dipole_alpha : float, optional
        Weighting of inhibitory synaptic current relative to excitatory current. Default 1.0.

    Returns
    -------
    tuple of (time_lfp_s, lfp_filtered_mv)
        Downsampled time array in seconds and filtered LFP signal.
    """
    e_ex = LIF_E_SYN_EX_MV  # 0.0 mV
    if cl_arr is not None:
        # Dynamic chloride reversal
        e_in = np.where(cl_arr > CHLORIDE_THRESHOLD, LIF_E_SYN_IN_DEP_MV, LIF_E_SYN_IN_NORMAL_MV)
    else:
        e_in = LIF_E_SYN_IN_NORMAL_MV

    # Transmembrane synaptic currents (I = g * (v - E_rev))
    # Excitatory inward current
    i_syn_e = ge_arr * (v_arr - e_ex)
    # Inhibitory outward / depolarizing current
    i_syn_i = gi_arr * (v_arr - e_in)

    # Net dipolar extracellular potential across pyramidal population
    raw_lfp = np.mean(i_syn_e + dipole_alpha * i_syn_i, axis=0)

    # Filtering & Resampling
    t_s = time_ms * 1e-3
    t_resampled, lfp_filtered = filter_and_resample_lfp(t_s, raw_lfp)
    return t_resampled, lfp_filtered


def filter_and_resample_lfp(
    t_s: np.ndarray,
    lfp_raw: np.ndarray,
    fs_target: float = LFP_SAMPLING_RATE_HZ,
    low_cut: float = LFP_FILTER_LOW_HZ,
    high_cut: float = LFP_FILTER_HIGH_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies Butterworth bandpass filtering (0.15 - 480 Hz) and resamples to target rate (1024 Hz).
    """
    dt_orig = t_s[1] - t_s[0] if len(t_s) > 1 else 1e-3
    fs_orig = 1.0 / dt_orig

    # 1. Bandpass Filter
    nyquist = 0.5 * fs_orig
    low = max(1e-4, low_cut / nyquist)
    high = min(0.99, high_cut / nyquist)

    if low < high:
        b, a = butter(4, [low, high], btype="band")
        lfp_filtered = filtfilt(b, a, lfp_raw)
    else:
        lfp_filtered = lfp_raw

    # 2. Resample to 1024 Hz
    duration_s = t_s[-1] - t_s[0]
    n_target = max(2, int(np.round(duration_s * fs_target)))
    lfp_resampled = resample(lfp_filtered, n_target)
    t_resampled = np.linspace(t_s[0], t_s[-1], n_target)

    return t_resampled, lfp_resampled
