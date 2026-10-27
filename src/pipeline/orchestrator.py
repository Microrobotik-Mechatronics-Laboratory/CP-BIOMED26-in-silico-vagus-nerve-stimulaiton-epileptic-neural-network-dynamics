"""
src/pipeline/orchestrator.py
==============================
End-to-end integration orchestrator connecting:
Level 1 (Axon NEURON) -> Bridge (Bundle) -> Level 2 (NTS Relay) -> Level 3 (Hippocampus) -> Level 4 (LFP/Metrics).
"""

from typing import Any, Optional
import numpy as np
from brian2 import Network, ms, mV, nS, defaultclock, start_scope

from config.defaults import (
    DEFAULT_BRIAN_DT_MS,
    DEFAULT_SIM_DURATION_MS,
    PATHOLOGY_PRESETS,
)
from src.pipeline.level1 import run_level1
from src.network.bridge import bundle_from_single_fiber
from src.network.nts_relay import build_nts_relay
from src.network.hippocampus import HippocampalFormation
from src.analysis.lfp import compute_lfp_from_states
from src.analysis.metrics import (
    compute_spectral_power,
    compute_firing_rate,
    compute_synchrony_index,
    compute_seizure_suppression_ratio,
)


def run_vns_epilepsy_trial(
    freq_hz: float = 20.0,
    amp_ma: float = 2.0,
    waveform: str = "rectangular",
    duration_ms: float = DEFAULT_SIM_DURATION_MS,
    condition: str = "epileptic",
    scl: Optional[float] = None,
    spr: Optional[float] = None,
    tau_cl_ms: Optional[float] = None,
    seed: int = 1000,
    record_lfp: bool = True,
    baseline_theta_power: Optional[float] = None,
) -> dict[str, Any]:
    """
    Executes a single end-to-end simulation trial.

    Parameters
    ----------
    freq_hz : float
        VNS frequency in Hz.
    amp_ma : float
        VNS amplitude in mA (0.0 for baseline / unstimulated).
    waveform : str
        "rectangular" (biphasic) or "dc" (monophasic).
    duration_ms : float
        Simulation duration in ms.
    condition : str
        "healthy" or "epileptic".
    scl, spr, tau_cl_ms : float, optional
        Custom pathology overrides.
    seed : int
        Random seed for Brian2 connectivity and jitter.
    record_lfp : bool
        Whether to record states for LFP calculation.
    baseline_theta_power : float, optional
        Baseline theta power to compute Seizure Suppression Ratio (SSR).

    Returns
    -------
    dict of Any
        Comprehensive trial results: firing rates, theta power, SSR, LFP time series.
    """
    start_scope()
    defaultclock.dt = DEFAULT_BRIAN_DT_MS * ms

    # 1. Pathology configuration
    preset = PATHOLOGY_PRESETS.get(condition, PATHOLOGY_PRESETS["epileptic"])
    scl_val = scl if scl is not None else preset["scl"]
    spr_val = spr if spr is not None else preset["spr"]
    tau_cl_val = tau_cl_ms if tau_cl_ms is not None else preset["tau_cl_ms"]

    # 2. Level 1: Peripheral Vagus Nerve stimulation (NEURON)
    if amp_ma > 0.0:
        distal_spikes = run_level1(
            freq_hz=freq_hz,
            amp_ma=amp_ma,
            duration_ms=duration_ms,
            waveform=waveform,
        )
    else:
        distal_spikes = []

    # 3. Bridge: Jittered fiber bundle
    bundle = bundle_from_single_fiber(
        distal_spikes,
        n_fibers=20,
        seed=seed,
    )

    # 4. Level 2: NTS Brainstem Relay
    nts, make_afferent_syn = build_nts_relay()
    syn_afferent = make_afferent_syn(bundle)

    # 5. Level 3: Epileptic Hippocampal Microcircuit
    hipp = HippocampalFormation(
        scl=scl_val,
        spr=spr_val,
        tau_cl_ms=tau_cl_val,
        seed=seed,
    )
    vns_synapses = hipp.attach_vns_input(nts)
    monitors = hipp.setup_monitors(record_lfp=record_lfp)

    # 6. Run simulation
    all_objs = [bundle, nts, syn_afferent] + hipp.all_network_objects()
    net = Network(*all_objs)
    net.run(duration_ms * ms)

    # 7. Extract Spiking Metrics
    duration_s = duration_ms * 1e-3
    results: dict[str, Any] = {
        "freq_hz": freq_hz,
        "amp_ma": amp_ma,
        "waveform": waveform,
        "duration_ms": duration_ms,
        "condition": condition,
        "scl": scl_val,
        "spr": spr_val,
        "tau_cl_ms": tau_cl_val,
        "seed": seed,
        "vns_spike_count": len(distal_spikes),
    }

    spike_trains_ca3: list[np.ndarray] = []
    for region in ["EC", "DG", "CA3", "CA1"]:
        spk_ex = monitors[f"spike_{region}_ex"]
        spk_in = monitors[f"spike_{region}_in"]
        n_ex = len(hipp.populations[region]["ex"])
        n_in = len(hipp.populations[region]["in"])

        fr_ex = compute_firing_rate(spk_ex.num_spikes, n_ex, duration_s)
        fr_in = compute_firing_rate(spk_in.num_spikes, n_in, duration_s)

        results[f"fr_{region}_ex"] = fr_ex
        results[f"fr_{region}_in"] = fr_in
        results[f"spikes_{region}_ex"] = spk_ex.num_spikes
        results[f"spike_t_{region}_ex"] = np.array(spk_ex.t / ms)
        results[f"spike_i_{region}_ex"] = np.array(spk_ex.i)
        results[f"spike_t_{region}_in"] = np.array(spk_in.t / ms)
        results[f"spike_i_{region}_in"] = np.array(spk_in.i)
        results[f"n_{region}_ex"] = n_ex
        results[f"n_{region}_in"] = n_in

        if region == "CA3":
            # Extract per-neuron spike trains for synchrony
            for i in range(n_ex):
                spike_trains_ca3.append(spk_ex.t[spk_ex.i == i] / ms)

    results["synchrony_ca3"] = compute_synchrony_index(spike_trains_ca3, duration_s)

    # 8. Extract LFP & Spectral Biomarkers
    if record_lfp and f"state_CA3_ex" in monitors:
        state_mon = monitors["state_CA3_ex"]
        t_ms = state_mon.t / ms
        v_mat = state_mon.v / mV
        ge_mat = state_mon.ge / nS
        gi_mat = state_mon.gi / nS

        t_lfp_s, lfp_filt = compute_lfp_from_states(
            v_arr=v_mat,
            ge_arr=ge_mat * 1e-9,
            gi_arr=gi_mat * 1e-9,
            time_ms=t_ms,
        )
        freqs, psd, theta_power = compute_spectral_power(lfp_filt, fs=1024.0)

        results["lfp_time_s"] = t_lfp_s
        results["lfp_signal"] = lfp_filt
        results["theta_power"] = theta_power
        results["psd_freqs"] = freqs
        results["psd_values"] = psd
        results["t_state_ms"] = t_ms
        if hasattr(state_mon, "Cl"):
            cl_ca3 = np.mean(state_mon.Cl, axis=0)
            cl_ca3_max = np.max(state_mon.Cl, axis=0)
            results["cl_ca3"] = cl_ca3
            results["cl_ca3_max"] = cl_ca3_max
            results["ein_ca3"] = -80.0 + 30.0 / (1.0 + np.exp(-50.0 * (cl_ca3 - 0.5)))
            results["ein_ca3_max"] = -80.0 + 30.0 / (1.0 + np.exp(-50.0 * (cl_ca3_max - 0.5)))

        if baseline_theta_power is not None:
            results["ssr"] = compute_seizure_suppression_ratio(theta_power, baseline_theta_power)
        else:
            results["ssr"] = 0.0
    else:
        results["theta_power"] = 0.0
        results["ssr"] = 0.0

    return results
