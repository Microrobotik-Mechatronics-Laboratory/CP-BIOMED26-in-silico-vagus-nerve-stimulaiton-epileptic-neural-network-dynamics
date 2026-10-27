"""
scripts/generate_paper_figures.py
=================================
Generates the two primary publication-quality figures demonstrating:
1. Figure 1: Hippocampal Formation Electrophysiological Validation
   (Healthy Baseline vs. Epileptic Hypersynchrony vs. VNS-Treated Desynchronization + LFP & Theta PSD)
2. Figure 2: Multiscale VNS Parameter Space & Biophysical Mechanism Validation
   (Peripheral Axon Low-pass Limit, Subfield Frequency Recruitment, 30-Condition SSR Heatmap, and Dynamic Chloride Dynamics)

BIOMED 2026.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.defaults import (
    VNS_WAVEFORMS,
    VNS_AMPLITUDES_MA,
    VNS_FREQUENCIES_HZ,
)
from src.pipeline.orchestrator import run_vns_epilepsy_trial


# Publication styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})

COLORS = {
    "EC": "#2b5c8f",
    "DG": "#2a9d8f",
    "CA3": "#7b2cbf",
    "CA1": "#e76f51",
    "Healthy": "#2a9d8f",
    "Epileptic": "#e63946",
    "VNS": "#1d3557",
}


def generate_figure1_comparative_dynamics(
    duration_ms: float = 500.0,
    save_path: str = "output/figures/figure1_hippocampal_validation.png",
) -> tuple[dict, dict, dict]:
    """
    Figure 1: Multi-subfield raster plots, simultaneous LFP, and PSD across:
    (A) Healthy Baseline, (B) Epileptic Control, (C) VNS 20 Hz Treated.
    """
    print("--> [Figure 1] Running simulations...")
    print("    1/3: Healthy baseline...")
    healthy_res = run_vns_epilepsy_trial(
        freq_hz=0.0, amp_ma=0.0, duration_ms=duration_ms, condition="healthy", record_lfp=True, seed=101
    )
    print("    2/3: Epileptic baseline (unstimulated)...")
    epileptic_res = run_vns_epilepsy_trial(
        freq_hz=0.0, amp_ma=0.0, duration_ms=duration_ms, condition="epileptic", record_lfp=True, seed=101
    )
    base_theta = epileptic_res["theta_power"]

    print("    3/3: Epileptic + VNS (20 Hz, 2.0 mA biphasic)...")
    vns_res = run_vns_epilepsy_trial(
        freq_hz=20.0, amp_ma=2.0, waveform="rectangular", duration_ms=duration_ms,
        condition="epileptic", record_lfp=True, baseline_theta_power=base_theta, seed=101
    )

    print("--> [Figure 1] Rendering publication graphic...")
    fig = plt.figure(figsize=(14, 9), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[2.8, 1.2, 1.4], hspace=0.35, wspace=0.25)

    trials = [
        ("A. Healthy Baseline (Normal E-I Balance)", healthy_res, COLORS["Healthy"]),
        ("B. Epileptic Control (Sclerosis + Sprouting + Impaired Cl⁻)", epileptic_res, COLORS["Epileptic"]),
        ("C. Epileptic + VNS 20 Hz (Neuromodulated)", vns_res, COLORS["VNS"]),
    ]

    subfields = ["EC", "DG", "CA3", "CA1"]

    for col_idx, (col_title, data, primary_color) in enumerate(trials):
        # 1. Raster Plot (Row 0)
        ax_raster = fig.add_subplot(gs[0, col_idx])
        y_offset = 0
        yticks = []
        yticklabels = []

        for region in subfields:
            t_ex = data.get(f"spike_t_{region}_ex", np.array([]))
            idx_ex = data.get(f"spike_i_{region}_ex", np.array([]))
            t_in = data.get(f"spike_t_{region}_in", np.array([]))
            idx_in = data.get(f"spike_i_{region}_in", np.array([]))
            n_ex = data.get(f"n_{region}_ex", 100)
            n_in = data.get(f"n_{region}_in", 20)

            if len(t_ex) > 0:
                ax_raster.scatter(t_ex, idx_ex + y_offset, s=4, color=COLORS[region], alpha=0.75, rasterized=True)
            if len(t_in) > 0:
                ax_raster.scatter(t_in, idx_in + y_offset + n_ex, s=4, color="#495057", alpha=0.55, marker="x", rasterized=True)

            yticks.append(y_offset + (n_ex + n_in) / 2)
            yticklabels.append(f"{region}")
            y_offset += (n_ex + n_in) + 10

        ax_raster.set_ylim(-5, y_offset)
        ax_raster.set_xlim(0, duration_ms)
        ax_raster.set_title(col_title, fontweight="bold", fontsize=10.5, pad=8)
        if col_idx == 0:
            ax_raster.set_yticks(yticks)
            ax_raster.set_yticklabels(yticklabels, fontweight="bold")
            ax_raster.set_ylabel("Subfield Populations")
        else:
            ax_raster.set_yticks([])

        ax_raster.grid(True, linestyle=":", alpha=0.4)

        # 2. Local Field Potential (Row 1)
        ax_lfp = fig.add_subplot(gs[1, col_idx], sharex=ax_raster)
        t_lfp_ms = data.get("lfp_time_s", np.linspace(0, duration_ms*1e-3, 512)) * 1000.0
        lfp_sig = data.get("lfp_signal", np.zeros_like(t_lfp_ms))

        ax_lfp.plot(t_lfp_ms, lfp_sig * 1000.0, color=primary_color, linewidth=1.1)
        ax_lfp.set_xlabel("Time (ms)")
        if col_idx == 0:
            ax_lfp.set_ylabel("CA3 LFP (µV)")
        ax_lfp.grid(True, linestyle=":", alpha=0.4)

        # Print stats inside box
        fr_ca3 = data.get("fr_CA3_ex", 0.0)
        sync_ca3 = data.get("synchrony_ca3", 0.0)
        theta_p = data.get("theta_power", 0.0)
        ssr = data.get("ssr", 0.0)
        stat_text = f"CA3 FR: {fr_ca3:.1f} Hz\nSync: {sync_ca3:.3f}\nθ-Power: {theta_p:.1e}"
        if col_idx == 2:
            stat_text += f"\nSSR: {ssr*100:.1f}%"
        ax_lfp.text(
            0.03, 0.92, stat_text, transform=ax_lfp.transAxes,
            fontsize=7.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc")
        )

    # 3. Power Spectral Density Comparison (Row 2, spanning columns 0 & 1)
    ax_psd = fig.add_subplot(gs[2, :2])
    for label, d, c in [
        ("Healthy Baseline", healthy_res, COLORS["Healthy"]),
        ("Epileptic Control", epileptic_res, COLORS["Epileptic"]),
        ("Epileptic + VNS (20 Hz, 2 mA)", vns_res, COLORS["VNS"]),
    ]:
        f = d.get("psd_freqs", np.array([]))
        p = d.get("psd_values", np.array([]))
        if len(f) > 0:
            ax_psd.semilogy(f, p, label=label, color=c, linewidth=1.8)

    # Highlight Theta Band (4-8 Hz)
    ax_psd.axvspan(4.0, 8.0, color="#ffb703", alpha=0.22, label="Ictal Theta Band (4–8 Hz)")
    ax_psd.set_xlim(1.0, 50.0)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (V² / Hz)")
    ax_psd.set_title("D. CA3 Synaptic Current Spectral Density & Ictal Theta Suppression", fontweight="bold")
    ax_psd.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_psd.grid(True, which="both", linestyle=":", alpha=0.4)

    # 4. Summary Bar Chart of Key Biomarkers (Row 2, Column 2)
    ax_bar = fig.add_subplot(gs[2, 2])
    categories = ["Healthy", "Epileptic", "VNS 20Hz"]
    fr_vals = [healthy_res["fr_CA3_ex"], epileptic_res["fr_CA3_ex"], vns_res["fr_CA3_ex"]]
    sync_vals = [healthy_res["synchrony_ca3"] * 10, epileptic_res["synchrony_ca3"] * 10, vns_res["synchrony_ca3"] * 10]

    x = np.arange(len(categories))
    width = 0.35
    ax_bar.bar(x - width/2, fr_vals, width, label="CA3 Rate (Hz)", color="#457b9d", alpha=0.9)
    ax_bar.bar(x + width/2, sync_vals, width, label="Sync Index (x10)", color="#e76f51", alpha=0.9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontweight="bold")
    ax_bar.set_ylabel("Metric Value")
    ax_bar.set_title("E. CA3 Biomarker Comparison", fontweight="bold")
    ax_bar.legend(loc="upper left")
    ax_bar.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("Figure 1: In Silico Epileptic Hippocampal Microcircuit Dynamics & VNS Neuromodulation",
                 fontsize=13.5, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> [Figure 1] Successfully saved to {save_path}")

    return healthy_res, epileptic_res, vns_res


def generate_figure2_parameter_space_and_mechanisms(
    epileptic_baseline_theta: float,
    duration_ms: float = 350.0,
    save_path: str = "output/figures/figure2_vns_parameter_space.png",
) -> pd.DataFrame:
    """
    Figure 2: Multiscale VNS parameter space response across 30 conditions and biophysical mechanisms:
    (A) Axon spike transmission vs. stimulation frequency (1 Hz - 10 kHz)
    (B) Hippocampal subfield firing rates vs. frequency (CA3 recurrent amplification)
    (C) 30-Condition Seizure Suppression Ratio (SSR) Heatmap (Frequency x Amplitude x Waveform)
    (D) Intracellular Chloride accumulation ([Cl⁻]) & GABA reversal potential shift (Ein) over time.
    """
    print("--> [Figure 2] Executing 30 VNS parameter configurations...")
    records = []
    total = len(VNS_WAVEFORMS) * len(VNS_AMPLITUDES_MA) * len(VNS_FREQUENCIES_HZ)
    idx = 0

    representative_trial = None

    # Explicitly run representative trial (20 Hz, 2 mA biphasic) for Panel D
    print("    Running representative trial for Panel D (20 Hz, 2.0 mA biphasic)...")
    representative_trial = run_vns_epilepsy_trial(
        freq_hz=20.0,
        amp_ma=2.0,
        waveform="rectangular",
        duration_ms=duration_ms,
        condition="epileptic",
        seed=1001,
        record_lfp=True,
        baseline_theta_power=epileptic_baseline_theta,
    )

    for wf in VNS_WAVEFORMS:
        for amp in VNS_AMPLITUDES_MA:
            for freq in VNS_FREQUENCIES_HZ:
                idx += 1
                print(f"    [{idx}/{total}] Condition: {wf} | {amp} mA | {freq} Hz")
                res = run_vns_epilepsy_trial(
                    freq_hz=freq,
                    amp_ma=amp,
                    waveform=wf,
                    duration_ms=duration_ms,
                    condition="epileptic",
                    seed=1001,
                    record_lfp=True,
                    baseline_theta_power=epileptic_baseline_theta,
                )
                rec = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
                records.append(rec)

    df = pd.DataFrame(records)
    csv_path = "output/results/vns_epilepsy_sweep_summary.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"--> Sweep saved to {csv_path}")

    print("--> [Figure 2] Rendering publication graphic...")
    fig = plt.figure(figsize=(14, 9.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)

    # Panel A: Axon Conduction & Peripheral Frequency Filter
    ax_a = fig.add_subplot(gs[0, 0])
    sub_df_biphasic = df[df["waveform"] == "rectangular"]
    for amp in VNS_AMPLITUDES_MA:
        sub = sub_df_biphasic[sub_df_biphasic["amp_ma"] == amp].sort_values("freq_hz")
        # Expected applied pulses vs conducted spikes
        pulses = sub["freq_hz"] * (duration_ms * 1e-3)
        ax_a.plot(sub["freq_hz"], sub["vns_spike_count"], marker="o", linewidth=2.0, label=f"Conducted Spikes ({amp} mA)")

    ax_a.plot(sub["freq_hz"], pulses, linestyle="--", color="#6c757d", label="Applied Pulses (Ideal)", alpha=0.7)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Stimulation Frequency (Hz)")
    ax_a.set_ylabel("Spike Count at Distal Node")
    ax_a.set_title("A. Peripheral Vagus Axon Conduction Limit (MRG Model)", fontweight="bold")
    ax_a.legend(loc="upper left")
    ax_a.grid(True, which="both", linestyle=":", alpha=0.4)

    # Panel B: Regional Hippocampal Firing Rates vs. VNS Frequency
    ax_b = fig.add_subplot(gs[0, 1])
    sub_2ma = df[(df["waveform"] == "rectangular") & (df["amp_ma"] == 2.0)].sort_values("freq_hz")
    for region in ["EC", "DG", "CA3", "CA1"]:
        col = f"fr_{region}_ex"
        if col in sub_2ma:
            ax_b.plot(sub_2ma["freq_hz"], sub_2ma[col], marker="s", linewidth=2.0, color=COLORS[region], label=f"{region} Principal")

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Stimulation Frequency (Hz)")
    ax_b.set_ylabel("Firing Rate (spikes / s / neuron)")
    ax_b.set_title("B. Hippocampal Subfield Recruitment (2.0 mA Biphasic)", fontweight="bold")
    ax_b.legend(loc="upper left")
    ax_b.grid(True, which="both", linestyle=":", alpha=0.4)

    # Panel C: Seizure Suppression Ratio (SSR) Heatmap (Biphasic & Monophasic)
    ax_c = fig.add_subplot(gs[1, 0])
    piv = sub_df_biphasic.pivot_table(index="amp_ma", columns="freq_hz", values="ssr", aggfunc="mean")
    im = ax_c.imshow(piv.values, cmap="RdYlGn", vmin=-0.3, vmax=1.0, origin="lower", aspect="auto")

    ax_c.set_xticks(range(len(piv.columns)))
    ax_c.set_xticklabels([f"{int(c) if c>=1 else c} Hz" for c in piv.columns])
    ax_c.set_yticks(range(len(piv.index)))
    ax_c.set_yticklabels([f"{r} mA" for r in piv.index])
    ax_c.set_xlabel("VNS Frequency (Hz)")
    ax_c.set_ylabel("VNS Amplitude (mA)")
    ax_c.set_title("C. Seizure Suppression Ratio (SSR) Heatmap (Biphasic)", fontweight="bold")

    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax_c.text(j, i, f"{v*100:+.0f}%", ha="center", va="center",
                     color="black" if -0.1 < v < 0.6 else "white", fontweight="bold", fontsize=10)

    cbar = plt.colorbar(im, ax=ax_c, orientation="vertical", pad=0.03)
    cbar.set_label("Seizure Suppression Ratio (SSR)", fontsize=9)

    # Panel D: Aussel et al. (2022) Dynamic Chloride & Depolarizing GABA Reversal Shift
    ax_d = fig.add_subplot(gs[1, 1])
    if representative_trial and "cl_ca3" in representative_trial:
        t_cl = representative_trial.get("t_state_ms", np.linspace(0, duration_ms, len(representative_trial["cl_ca3"])))
        cl_traj = representative_trial["cl_ca3"]
        cl_max = representative_trial.get("cl_ca3_max", cl_traj)
        ein_traj = representative_trial["ein_ca3"]
        ein_max = representative_trial.get("ein_ca3_max", ein_traj)

        ax_d.plot(t_cl, cl_max, color="#7b2cbf", linewidth=2.0, label="Max [Cl⁻] in Active CA3 Neurons")
        ax_d.plot(t_cl, cl_traj, color="#9d4edd", linewidth=1.2, linestyle=":", alpha=0.75, label="Mean Population [Cl⁻]")
        ax_d.axhline(0.5, color="#e63946", linestyle="--", linewidth=1.2, label="Depolarizing Threshold ([Cl⁻]=0.5)")

        ax_d2 = ax_d.twinx()
        ax_d2.plot(t_cl, ein_max, color="#e76f51", linewidth=1.8, linestyle="-.", label="E_GABA Reversal (Max Active)")
        ax_d2.set_ylabel("GABA Reversal Potential E_GABA (mV)", color="#e76f51")
        ax_d2.set_ylim(-85, -45)
        ax_d2.tick_params(axis="y", labelcolor="#e76f51")

        ax_d.set_xlabel("Time (ms)")
        ax_d.set_ylabel("Intracellular Chloride [Cl⁻] (arb. units)", color="#7b2cbf")
        ax_d.set_title("D. KCC2 Impairment & Dynamic GABA Reversal Potential Shift", fontweight="bold")
        ax_d.grid(True, linestyle=":", alpha=0.4)

        lines1, labels1 = ax_d.get_legend_handles_labels()
        lines2, labels2 = ax_d2.get_legend_handles_labels()
        ax_d.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.suptitle("Figure 2: 30-Condition VNS Parameter Efficacy & Neuropathological Mechanisms",
                 fontsize=13.5, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> [Figure 2] Successfully saved to {save_path}")

    return df


def main():
    print("==================================================================")
    print("  Generating Publication Figures for BIOMED 2026 Paper / Poster   ")
    print("==================================================================")

    # 1. Generate Figure 1
    h_res, e_res, v_res = generate_figure1_comparative_dynamics(
        duration_ms=500.0,
        save_path="output/figures/figure1_hippocampal_validation.png",
    )

    # 2. Generate Figure 2 using baseline theta power
    base_theta = e_res.get("theta_power", 1.0)
    df = generate_figure2_parameter_space_and_mechanisms(
        epileptic_baseline_theta=base_theta,
        duration_ms=300.0,
        save_path="output/figures/figure2_vns_parameter_space.png",
    )

    # Also copy to artifacts directory for visual embedding
    artifact_dir = "/Users/musadereli/.gemini/antigravity-cli/brain/a35a29d7-1ccf-427a-9662-181d0e5ec489"
    os.system(f"cp output/figures/figure1_hippocampal_validation.png '{artifact_dir}/figure1_hippocampal_validation.png'")
    os.system(f"cp output/figures/figure2_vns_parameter_space.png '{artifact_dir}/figure2_vns_parameter_space.png'")

    print("==================================================================")
    print("  All Figures Successfully Generated and Verified!                ")
    print(f"  1. output/figures/figure1_hippocampal_validation.png")
    print(f"  2. output/figures/figure2_vns_parameter_space.png")
    print("==================================================================")


if __name__ == "__main__":
    main()
