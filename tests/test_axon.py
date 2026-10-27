"""
tests/test_axon.py
==================
Unit tests for MRG axon stimulation and distal spike extraction.
"""

import pytest
from src.pipeline.level1 import run_level1


def test_axon_biphasic_stimulation():
    """Verify that a 20 Hz, 1.0 mA biphasic stimulus evokes propagating action potentials."""
    spikes = run_level1(freq_hz=20.0, amp_ma=1.0, duration_ms=100.0, waveform="rectangular")
    assert isinstance(spikes, list)
    assert len(spikes) >= 1, "Expected at least 1 spike to propagate to the distal node"
    # At 20 Hz in 100 ms, ~2 pulses are delivered
    assert len(spikes) <= 4


def test_axon_dc_stimulation():
    """Verify monophasic DC stimulation evokes spikes."""
    spikes = run_level1(freq_hz=20.0, amp_ma=1.0, duration_ms=100.0, waveform="dc")
    assert isinstance(spikes, list)
    assert len(spikes) >= 1


def test_axon_subthreshold_stimulation():
    """Verify that very small subthreshold stimulus (e.g. 0.001 mA) produces zero spikes."""
    spikes = run_level1(freq_hz=20.0, amp_ma=0.001, duration_ms=100.0)
    assert len(spikes) == 0, "Subthreshold stimulation should not evoke spikes"
