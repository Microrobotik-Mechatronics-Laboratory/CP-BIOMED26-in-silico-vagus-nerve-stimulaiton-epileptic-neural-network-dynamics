"""
tests/test_nts.py
=================
Unit tests for NTS relay and Tsodyks-Markram synaptic plasticity calibration.
"""

import numpy as np
import pytest
from brian2 import Network, SpikeMonitor, ms, defaultclock, start_scope

from config.defaults import DEFAULT_BRIAN_DT_MS
from src.network.bridge import bundle_from_single_fiber
from src.network.nts_relay import build_nts_relay


def test_nts_relay_activation():
    """Verify that afferent spikes drive NTS relay neurons."""
    start_scope()
    defaultclock.dt = DEFAULT_BRIAN_DT_MS * ms

    # Create 10 afferent spikes at 20 Hz (interval 50 ms)
    spike_times = [10.0, 60.0, 110.0, 160.0, 210.0]
    bundle = bundle_from_single_fiber(spike_times, n_fibers=20, jitter_ms=0.5, seed=42)

    nts, make_synapses = build_nts_relay(n_neurons=30)
    syn = make_synapses(bundle)

    mon = SpikeMonitor(nts)
    net = Network(bundle, nts, syn, mon)
    net.run(250.0 * ms)

    assert mon.num_spikes > 0, "NTS relay should fire in response to afferent bundle drive"
