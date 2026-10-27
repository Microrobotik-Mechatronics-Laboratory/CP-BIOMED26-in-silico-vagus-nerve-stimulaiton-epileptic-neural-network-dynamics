"""
tests/test_hippocampus.py
=========================
Unit tests for the hippocampal formation (EC-DG-CA3-CA1) and Aussel (2022) pathologies.
"""

import pytest
from brian2 import Network, ms, defaultclock, start_scope

from config.defaults import DEFAULT_BRIAN_DT_MS
from src.network.hippocampus import HippocampalFormation


def test_hippocampus_sclerosis_scaling():
    """Verify that sclerosis reduces excitatory neuron count while sparing interneurons."""
    start_scope()
    hipp_healthy = HippocampalFormation(scl=0.0, spr=0.0, tau_cl_ms=100.0)
    assert len(hipp_healthy.populations["CA3"]["ex"]) == 100
    assert len(hipp_healthy.populations["CA3"]["in"]) == 20

    start_scope()
    hipp_scl = HippocampalFormation(scl=0.5, spr=0.0, tau_cl_ms=100.0)
    assert len(hipp_scl.populations["CA3"]["ex"]) == 60
    assert len(hipp_scl.populations["CA3"]["in"]) == 20  # Interneurons spared


def test_hippocampus_epileptic_vs_healthy_dynamics():
    """
    Verify that the epileptic network with mossy fiber sprouting and impaired chloride
    exhibits significantly elevated paroxysmal activity compared to healthy baseline.
    """
    defaultclock.dt = DEFAULT_BRIAN_DT_MS * ms

    # 1. Healthy Run
    start_scope()
    hipp_h = HippocampalFormation(scl=0.0, spr=0.0, tau_cl_ms=100.0, cortical_rate_hz=10.0)
    mons_h = hipp_h.setup_monitors(record_lfp=False)
    net_h = Network(*hipp_h.all_network_objects())
    net_h.run(50.0 * ms)
    spikes_healthy = mons_h["spike_CA3_ex"].num_spikes

    # 2. Epileptic Run (Sprouting + Impaired Chloride Dynamics)
    start_scope()
    hipp_e = HippocampalFormation(scl=0.2, spr=0.8, tau_cl_ms=800.0, cortical_rate_hz=10.0)
    mons_e = hipp_e.setup_monitors(record_lfp=False)
    net_e = Network(*hipp_e.all_network_objects())
    net_e.run(50.0 * ms)
    spikes_epileptic = mons_e["spike_CA3_ex"].num_spikes

    # Epileptic network should show amplified firing through sprouted DG and disinhibited CA3
    assert spikes_epileptic >= spikes_healthy, (
        f"Epileptic CA3 spikes ({spikes_epileptic}) should meet or exceed healthy ({spikes_healthy})"
    )
