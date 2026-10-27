"""
src/visualization/plots.py
===========================
Publication-ready visualization generators:
- Multi-population raster plots
- LFP time series and power spectrograms
- Regional firing rates vs frequency
- Seizure Suppression Ratio (SSR) response surface heatmaps
"""

from typing import Any, Optional
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raster_and_lfp(
    trial_data: dict[str, Any],
    title: str = "Hippocampal Formation Dynamics",
    save_path: Optional[str] = "output/figures/raster_and_lfp.png",
) -> None:
    """
    Plots subfield firing raster along with the simultaneous local field potential (LFP).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Regional colors
    colors = {"EC": "#377eb8", "DG": "#4daf4a", "CA3": "#984ea3", "CA1": "#e41a1c"}

    # Raster plot
    y_offset = 0
    ticks = []
    labels = []
    for region in ["EC", "DG", "CA3", "CA1"]:
        # Excitatory
        n_ex = int(trial_data.get(f"spikes_{region}_ex", 0))
        # Note: if spike times are available
        ticks.append(y_offset + 50)
        labels.append(f"{region}")
        y_offset += 100

    ax1.set_ylabel("Subfields")
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(labels)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # LFP trace
    t_lfp = trial_data.get("lfp_time_s", np.linspace(0, 1, 1024))
    lfp_sig = trial_data.get("lfp_signal", np.zeros_like(t_lfp))

    ax2.plot(t_lfp * 1000.0, lfp_sig, color="#e41a1c", linewidth=1.2, label="CA3 LFP (mV)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("LFP (mV)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_frequency_response(
    df: pd.DataFrame,
    waveform: str = "rectangular",
    amp_ma: float = 2.0,
    save_path: Optional[str] = "output/figures/frequency_response.png",
) -> None:
    """
    Plots regional firing rates vs stimulation frequency (1 Hz - 10 kHz).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sub_df = df[(df["waveform"] == waveform) & (df["amp_ma"] == amp_ma)]
    if sub_df.empty:
        return

    grouped = sub_df.groupby("freq_hz").mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = grouped["freq_hz"]

    colors = {"EC": "#377eb8", "DG": "#4daf4a", "CA3": "#984ea3", "CA1": "#e41a1c"}
    for region in ["EC", "DG", "CA3", "CA1"]:
        col = f"fr_{region}_ex"
        if col in grouped:
            ax.plot(freqs, grouped[col], marker="o", linewidth=2.0, color=colors[region], label=f"{region} Excitatory")

    ax.set_xscale("log")
    ax.set_xlabel("Stimulation Frequency (Hz)")
    ax.set_ylabel("Firing Rate (spikes / s / neuron)")
    ax.set_title(f"Regional Frequency Response ({waveform.capitalize()}, {amp_ma} mA)", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_seizure_suppression_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = "output/figures/seizure_suppression_heatmap.png",
) -> None:
    """
    Plots Seizure Suppression Ratio (SSR) heatmaps for Monophasic DC vs Biphasic Rectangular waveforms.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    waveforms = [wf for wf in ["dc", "rectangular"] if wf in df["waveform"].values]
    if not waveforms:
        return

    fig, axes = plt.subplots(1, len(waveforms), figsize=(6 * len(waveforms), 5), sharey=True)
    if len(waveforms) == 1:
        axes = [axes]

    for ax, wf in zip(axes, waveforms):
        sub_df = df[df["waveform"] == wf]
        piv = sub_df.pivot_table(index="amp_ma", columns="freq_hz", values="ssr", aggfunc="mean")

        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.5, vmax=1.0, origin="lower", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{int(c) if c>=1 else c} Hz" for c in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r} mA" for r in piv.index])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (mA)")
        ax.set_title(f"Waveform: {wf.upper()} (Suppression Ratio)", fontweight="bold")

        # Overlay text values
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val)<0.6 else "white", fontsize=9)

    fig.colorbar(im, ax=axes, orientation="vertical", label="Seizure Suppression Ratio (SSR)")
    plt.suptitle("VNS Parameter Space: Seizure Suppression Efficacy", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
