"""
tests/test_lfp.py
=================
Unit tests for LFP simulation, filtering, and spectral analysis metrics.
"""

import numpy as np
import pytest

from src.analysis.lfp import compute_lfp_from_states, filter_and_resample_lfp
from src.analysis.metrics import (
    compute_spectral_power,
    compute_seizure_suppression_ratio,
    compute_firing_rate,
)


def test_filter_and_resample_lfp():
    """Verify Butterworth filtering and resampling to 1024 Hz."""
    t_s = np.linspace(0.0, 1.0, 2000)
    # 6 Hz theta sine wave + 500 Hz high frequency noise
    sig = np.sin(2 * np.pi * 6.0 * t_s) + 0.5 * np.sin(2 * np.pi * 500.0 * t_s)

    t_res, lfp_filt = filter_and_resample_lfp(t_s, sig, fs_target=1024.0)
    assert len(t_res) == 1024
    assert len(lfp_filt) == 1024
    # Bandpass should preserve 6 Hz power
    freqs, psd, band_power = compute_spectral_power(lfp_filt, fs=1024.0, band=(4.0, 8.0))
    assert band_power > 0.05


def test_seizure_suppression_ratio():
    """Verify SSR calculation for full, partial, and negative suppression."""
    # Baseline power = 100.0, VNS power = 20.0 -> SSR = 0.8 (80% suppression)
    ssr = compute_seizure_suppression_ratio(20.0, 100.0)
    assert pytest.approx(ssr, 0.01) == 0.8

    # Baseline = 100.0, VNS power = 0.0 -> SSR = 1.0 (complete suppression)
    ssr_full = compute_seizure_suppression_ratio(0.0, 100.0)
    assert pytest.approx(ssr_full, 0.01) == 1.0

    # VNS exacerbates power -> SSR < 0
    ssr_worse = compute_seizure_suppression_ratio(150.0, 100.0)
    assert ssr_worse < 0.0
