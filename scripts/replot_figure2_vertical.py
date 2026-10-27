"""
scripts/replot_figure2_vertical.py
==================================
Replots Figure 2 in a vertical (all panels stacked) layout with exact 45:55 aspect ratio:
Width : Height = 45 : 55 (e.g., 9.0 in x 11.0 in, 2700 x 3300 px @ 300 DPI).

Uses precomputed sweep data from output/results/vns_epilepsy_sweep_summary.csv and runs
only the single representative trial for Panel D chloride & GABA dynamics.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.defaults import VNS_AMPLITUDES_MA
from src.pipeline.orchestrator import run_vns_epilepsy_trial

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "EC": "#2b5c8f",
    "DG": "#2a9d8f",
    "CA3": "#7b2cbf",
    "CA1": "#e76f51",
}


def replot_figure2(
    csv_path: str = "output/results/vns_epilepsy_sweep_summary.csv",
    save_path: str = "output/figures/figure2_vns_parameter_space.png",
    duration_ms: float = 300.0,
) -> None:
    print(f"--> Loading precomputed sweep from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("--> Running representative trial for Panel D (20 Hz, 2.0 mA biphasic)...")
    rep_res = run_vns_epilepsy_trial(
        freq_hz=20.0,
        amp_ma=2.0,
        waveform="rectangular",
        duration_ms=duration_ms,
        condition="epileptic",
        seed=1001,
        record_lfp=True,
    )

    print("--> Rendering Figure 2 with vertical layout (45:55 aspect ratio)...")
    # Aspect ratio: 45:55 -> width : height = 9.0 in : 11.0 in
    width_in = 9.0
    height_in = width_in * (55.0 / 45.0)  # exactly 11.0 in

    fig = plt.figure(figsize=(width_in, height_in), dpi=300)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.05, 1.0], hspace=0.38)

    # -------------------------------------------------------------------------
    # Panel A: Axon Conduction Limit
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    sub_df_biphasic = df[df["waveform"] == "rectangular"]
    for amp in VNS_AMPLITUDES_MA:
        sub = sub_df_biphasic[sub_df_biphasic["amp_ma"] == amp].sort_values("freq_hz")
        ax_a.plot(sub["freq_hz"], sub["vns_spike_count"], marker="o", linewidth=1.8, label=f"Conducted Spikes ({amp} mA)")

    # Applied pulses line
    pulses = sub["freq_hz"] * (duration_ms * 1e-3)
    ax_a.plot(sub["freq_hz"], pulses, linestyle="--", color="#6c757d", label="Applied Pulses (Ideal)", alpha=0.75)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Stimulation Frequency (Hz)")
    ax_a.set_ylabel("Spike Count at Distal Node")
    ax_a.set_title("A. Peripheral Vagus Axon Conduction Limit (MRG Formalism, 15 Nodes)", fontweight="bold")
    ax_a.legend(loc="upper left", ncol=2)
    ax_a.grid(True, which="both", linestyle=":", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel B: Regional Hippocampal Firing Rates
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    sub_2ma = df[(df["waveform"] == "rectangular") & (df["amp_ma"] == 2.0)].sort_values("freq_hz")
    for region in ["EC", "DG", "CA3", "CA1"]:
        col = f"fr_{region}_ex"
        if col in sub_2ma:
            ax_b.plot(sub_2ma["freq_hz"], sub_2ma[col], marker="s", linewidth=1.8, color=COLORS[region], label=f"{region} Principal")

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Stimulation Frequency (Hz)")
    ax_b.set_ylabel("Firing Rate (spikes/s/neuron)")
    ax_b.set_title("B. Regional Hippocampal Recruitment & CA3 Amplification (2.0 mA Biphasic)", fontweight="bold")
    ax_b.legend(loc="upper left", ncol=4)
    ax_b.grid(True, which="both", linestyle=":", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel C: Seizure Suppression Ratio (SSR) Heatmap
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2, 0])
    piv = sub_df_biphasic.pivot_table(index="amp_ma", columns="freq_hz", values="ssr", aggfunc="mean")
    im = ax_c.imshow(piv.values, cmap="RdYlGn", vmin=-0.5, vmax=1.0, origin="lower", aspect="auto")

    ax_c.set_xticks(range(len(piv.columns)))
    ax_c.set_xticklabels([f"{int(c) if c>=1 else c} Hz" for c in piv.columns])
    ax_c.set_yticks(range(len(piv.index)))
    ax_c.set_yticklabels([f"{r} mA" for r in piv.index])
    ax_c.set_xlabel("VNS Frequency (Hz)")
    ax_c.set_ylabel("VNS Amplitude (mA)")
    ax_c.set_title("C. Seizure Suppression Ratio (SSR) Across Parameter Space (Biphasic Waveform)", fontweight="bold")

    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax_c.text(j, i, f"{v*100:+.0f}%", ha="center", va="center",
                      color="black" if -0.1 < v < 0.6 else "white", fontweight="bold", fontsize=10)

    cbar = plt.colorbar(im, ax=ax_c, orientation="vertical", pad=0.02)
    cbar.set_label("Seizure Suppression Ratio (SSR)", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel D: Intracellular Chloride & GABA Reversal Potential Shift
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[3, 0])
    t_cl = rep_res.get("t_state_ms", np.linspace(0, duration_ms, len(rep_res["cl_ca3"])))
    cl_traj = rep_res["cl_ca3"]
    cl_max = rep_res.get("cl_ca3_max", cl_traj)
    ein_max = rep_res.get("ein_ca3_max", rep_res["ein_ca3"])

    l1 = ax_d.plot(t_cl, cl_max, color="#7b2cbf", linewidth=1.8, label="Max [Cl⁻] in Active CA3 Neurons")
    l2 = ax_d.plot(t_cl, cl_traj, color="#9d4edd", linewidth=1.2, linestyle=":", alpha=0.8, label="Mean Population [Cl⁻]")
    l3 = ax_d.axhline(0.5, color="#e63946", linestyle="--", linewidth=1.2, label="Depolarizing Threshold ([Cl⁻]=0.5)")

    ax_d2 = ax_d.twinx()
    l4 = ax_d2.plot(t_cl, ein_max, color="#e76f51", linewidth=1.8, linestyle="-.", label="E_GABA Reversal (Max Active)")
    ax_d2.set_ylabel("GABA Reversal E_GABA (mV)", color="#e76f51")
    ax_d2.set_ylim(-85, -45)
    ax_d2.tick_params(axis="y", labelcolor="#e76f51")

    ax_d.set_xlabel("Time (ms)")
    ax_d.set_ylabel("Chloride [Cl⁻] (arb. units)", color="#7b2cbf")
    ax_d.set_title("D. KCC2 Impairment & Dynamic GABA Reversal Potential Shift (Aussel et al., 2022)", fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.4)

    lines = l1 + l2 + [l3] + l4
    labels = [l.get_label() for l in lines]
    ax_d.legend(lines, labels, loc="upper left", ncol=2, fontsize=8)

    plt.suptitle("Figure 2: 30-Condition VNS Parameter Efficacy & Neuropathological Mechanisms",
                 fontsize=12, fontweight="bold", y=0.995)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also update artifact copy
    artifact_dir = "/Users/musadereli/.gemini/antigravity-cli/brain/a35a29d7-1ccf-427a-9662-181d0e5ec489"
    os.system(f"cp '{save_path}' '{artifact_dir}/figure2_vns_parameter_space.png'")

    print(f"--> [Done] Saved vertically stacked Figure 2 (45:55 ratio) to {save_path} and {artifact_dir}")


if __name__ == "__main__":
    replot_figure2()
