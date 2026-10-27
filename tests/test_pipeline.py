"""
tests/test_pipeline.py
======================
Integration tests for the complete VNS -> NTS -> Epileptic Hippocampus pipeline.
"""

import pytest
from src.pipeline.orchestrator import run_vns_epilepsy_trial


def test_vns_epilepsy_trial_execution():
    """Verify end-to-end trial runs and returns required biomarkers."""
    res = run_vns_epilepsy_trial(
        freq_hz=20.0,
        amp_ma=2.0,
        waveform="rectangular",
        duration_ms=50.0,
        condition="epileptic",
        record_lfp=True,
    )

    assert "vns_spike_count" in res
    assert "fr_CA3_ex" in res
    assert "synchrony_ca3" in res
    assert "theta_power" in res
    assert "lfp_signal" in res
    assert len(res["lfp_signal"]) > 0
    assert res["vns_spike_count"] > 0
