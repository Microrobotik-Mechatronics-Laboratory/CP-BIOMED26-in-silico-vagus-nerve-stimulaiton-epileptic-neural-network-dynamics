"""
main.py
=======
Command Line Interface (CLI) for In Silico VNS Modulation of Epileptic Hippocampal Dynamics.
BIOMED 2026.
"""

import argparse
import sys
import os
import pandas as pd

from config.defaults import (
    DEFAULT_SIM_DURATION_MS,
    VNS_WAVEFORMS,
    VNS_AMPLITUDES_MA,
    VNS_FREQUENCIES_HZ,
)
from src.pipeline.orchestrator import run_vns_epilepsy_trial
from src.sweep.vns_epilepsy_sweep import run_full_vns_sweep
from src.visualization.plots import (
    plot_raster_and_lfp,
    plot_frequency_response,
    plot_seizure_suppression_heatmap,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="In Silico VNS Modulation of Epileptic Neural Network Dynamics (BIOMED 2026)"
    )
    parser.add_argument(
        "--run-sweep",
        action="store_true",
        help="Execute the full 30 VNS parameter configurations x 5 repeats sweep.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=20.0,
        help="Stimulation frequency in Hz (default: 20.0 Hz).",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=2.0,
        help="Stimulation amplitude in mA (default: 2.0 mA).",
    )
    parser.add_argument(
        "--waveform",
        type=str,
        default="rectangular",
        choices=["rectangular", "dc"],
        help="Stimulus waveform ('rectangular' for biphasic, 'dc' for monophasic).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_SIM_DURATION_MS,
        help="Simulation duration in ms (default: 1000 ms).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="epileptic",
        choices=["healthy", "epileptic"],
        help="Network pathology state ('healthy' or 'epileptic').",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeats per sweep condition (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save sweep results and generated figures.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate analysis figures from sweep or single trial results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    results_dir = os.path.join(args.output_dir, "results")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if args.run_sweep:
        print("=================================================================")
        print("  Running 30 VNS Configurations Sweep on Epileptic Hippocampus   ")
        print("=================================================================")
        df = run_full_vns_sweep(
            waveforms=VNS_WAVEFORMS,
            amplitudes_ma=VNS_AMPLITUDES_MA,
            frequencies_hz=VNS_FREQUENCIES_HZ,
            n_repeats=args.repeats,
            duration_ms=args.duration,
            output_dir=results_dir,
            verbose=True,
        )
        if args.plot or True:  # Automatically generate summary figures on sweep
            print("--> Generating publication figures...")
            plot_frequency_response(
                df,
                waveform="rectangular",
                amp_ma=2.0,
                save_path=os.path.join(figures_dir, "frequency_response_biphasic_2ma.png"),
            )
            plot_seizure_suppression_heatmap(
                df,
                save_path=os.path.join(figures_dir, "seizure_suppression_heatmap.png"),
            )
            print(f"--> Done! Figures saved in {figures_dir}")
        return

    # Single Trial Mode
    print(f"--> Running Single Trial: {args.condition.upper()} | {args.waveform} | {args.amp} mA | {args.freq} Hz")
    res = run_vns_epilepsy_trial(
        freq_hz=args.freq,
        amp_ma=args.amp,
        waveform=args.waveform,
        duration_ms=args.duration,
        condition=args.condition,
        record_lfp=True,
    )
    print("-----------------------------------------------------------------")
    print(f"  VNS Evoked Axon Spikes: {res.get('vns_spike_count', 0)}")
    print(f"  CA3 Excitatory Firing Rate: {res.get('fr_CA3_ex', 0.0):.2f} Hz")
    print(f"  CA3 Synchrony Index: {res.get('synchrony_ca3', 0.0):.4f}")
    print(f"  Theta Band (4-8 Hz) Power: {res.get('theta_power', 0.0):.4e}")
    print("-----------------------------------------------------------------")

    if args.plot:
        out_fig = os.path.join(figures_dir, f"trial_{args.condition}_{args.waveform}_{args.freq}Hz.png")
        plot_raster_and_lfp(res, title=f"Trial: {args.condition.capitalize()} ({args.freq} Hz, {args.amp} mA)", save_path=out_fig)
        print(f"--> Figure saved: {out_fig}")


if __name__ == "__main__":
    main()
