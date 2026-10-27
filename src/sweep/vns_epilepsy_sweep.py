"""
src/sweep/vns_epilepsy_sweep.py
================================
Systematic sweep engine executing 30 VNS parameter configurations x 5 repeats
on the epileptogenic human hippocampal formation.
"""

from typing import Any, Optional
import os
import pandas as pd
import numpy as np

from config.defaults import (
    VNS_WAVEFORMS,
    VNS_AMPLITUDES_MA,
    VNS_FREQUENCIES_HZ,
    DEFAULT_N_REPEATS,
    DEFAULT_BASE_SEED,
    DEFAULT_SIM_DURATION_MS,
)
from src.pipeline.orchestrator import run_vns_epilepsy_trial


def run_full_vns_sweep(
    waveforms: list[str] = VNS_WAVEFORMS,
    amplitudes_ma: list[float] = VNS_AMPLITUDES_MA,
    frequencies_hz: list[float] = VNS_FREQUENCIES_HZ,
    n_repeats: int = DEFAULT_N_REPEATS,
    duration_ms: float = DEFAULT_SIM_DURATION_MS,
    output_dir: str = "output/results",
    record_lfp: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Runs the 30 VNS conditions (2 waveforms x 3 amplitudes x 5 frequencies) x n_repeats.

    Parameters
    ----------
    waveforms : list of str
    amplitudes_ma : list of float
    frequencies_hz : list of float
    n_repeats : int
    duration_ms : float
    output_dir : str
    record_lfp : bool
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Table of all simulation trial metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Run Baseline Unstimulated Epileptic Control (0 mA)
    if verbose:
        print("--> Running Baseline Epileptic Control (0 mA)...")
    baseline_trial = run_vns_epilepsy_trial(
        freq_hz=0.0,
        amp_ma=0.0,
        waveform="none",
        duration_ms=duration_ms,
        condition="epileptic",
        record_lfp=record_lfp,
    )
    baseline_theta_power = baseline_trial.get("theta_power", 1.0)
    if baseline_theta_power <= 0:
        baseline_theta_power = 1.0

    if verbose:
        print(f"    Baseline Theta Power (4-8 Hz): {baseline_theta_power:.4e}")
        print(f"    Baseline CA3 Firing Rate: {baseline_trial.get('fr_CA3_ex', 0.0):.2f} Hz")

    records: list[dict[str, Any]] = []

    # Store baseline record
    base_rec = {k: v for k, v in baseline_trial.items() if not isinstance(v, np.ndarray)}
    base_rec["ssr"] = 0.0
    records.append(base_rec)

    total_conditions = len(waveforms) * len(amplitudes_ma) * len(frequencies_hz)
    cond_idx = 0

    # 2. Sweep across all 30 conditions
    for wf in waveforms:
        for amp in amplitudes_ma:
            for freq in frequencies_hz:
                cond_idx += 1
                if verbose:
                    print(f"--> [{cond_idx}/{total_conditions}] Condition: {wf}, {amp} mA, {freq} Hz")

                for rep in range(n_repeats):
                    seed = DEFAULT_BASE_SEED + rep
                    trial_res = run_vns_epilepsy_trial(
                        freq_hz=freq,
                        amp_ma=amp,
                        waveform=wf,
                        duration_ms=duration_ms,
                        condition="epileptic",
                        seed=seed,
                        record_lfp=record_lfp,
                        baseline_theta_power=baseline_theta_power,
                    )
                    # Exclude large raw arrays from CSV table
                    clean_rec = {
                        k: v for k, v in trial_res.items()
                        if not isinstance(v, np.ndarray)
                    }
                    clean_rec["repeat"] = rep
                    records.append(clean_rec)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "vns_epilepsy_sweep.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"--> Sweep completed! Saved {len(df)} records to {csv_path}")

    return df
