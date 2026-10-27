"""
src/stim/extracellular_field.py
================================
Extracellular field calculations and stimulation waveform generators.
"""

from typing import Any
import numpy as np

# rho [Ω·cm] * I [µA] / r [cm] yields Ω·µA = 1e-6 V = 1e-3 mV.
_UV_TO_MV: float = 1e-3


def point_source_potential(
    x: float, y: float, z: float,
    elec_pos: tuple[float, float, float],
    current: float, rho: float = 300.0,
) -> float:
    """
    Computes extracellular potential from a point-source electrode (McNeal 1976 / Rattay 1986).

    Parameters
    ----------
    x, y, z : float
        Coordinates of the target section (µm).
    elec_pos : tuple of float
        Electrode position (ex, ey, ez) in µm.
    current : float
        Injected current in µA.
    rho : float, optional
        Resistivity of homogeneous medium (Ω·cm). Default 300.0.

    Returns
    -------
    float
        Extracellular potential in mV.
    """
    ex, ey, ez = elec_pos
    r_um = np.sqrt((x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2)
    r_cm = np.maximum(r_um * 1e-4, 1e-6)  # Convert µm to cm, prevent singularity at r=0
    return _UV_TO_MV * (rho * current) / (4.0 * np.pi * r_cm)


def biphasic_waveform(
    freq_hz: float, amp: float, duration_ms: float, dt: float = 0.005,
    pulse_width_ms: float = 0.1, waveform: str = "rectangular",
    interphase_gap_ms: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates monophasic (dc) or charge-balanced biphasic (rectangular) pulse trains.

    Parameters
    ----------
    freq_hz : float
        Stimulation frequency in Hz.
    amp : float
        Current amplitude in µA.
    duration_ms : float
        Total duration in ms.
    dt : float, optional
        Time step in ms. Default 0.005 ms.
    pulse_width_ms : float, optional
        Duration of each pulse phase in ms. Default 0.1 ms.
    waveform : str, optional
        "rectangular" (charge-balanced biphasic) or "dc" (monophasic). Default "rectangular".
    interphase_gap_ms : float, optional
        Gap between phases in ms. Default 0.05 ms.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Time array (ms) and current waveform array (µA).
    """
    t_vec = np.arange(0, duration_ms, dt)
    i_vec = np.zeros_like(t_vec)

    period_ms = 1000.0 / freq_hz

    # Limit pulse width and gap at high frequencies to prevent pulse overlap
    pw = min(pulse_width_ms, period_ms / 4.0)
    gap = min(interphase_gap_ms, period_ms / 8.0)

    for t_start in np.arange(0, duration_ms, period_ms):
        # Phase 1: Cathodic / stimulus pulse
        mask1 = (t_vec >= t_start) & (t_vec < t_start + pw)
        i_vec[mask1] = amp

        if waveform == "dc":
            continue

        # Phase 2: Anodic charge-balancing phase
        neg_start = t_start + pw + gap
        mask2 = (t_vec >= neg_start) & (t_vec < neg_start + pw)
        i_vec[mask2] = -amp

    return t_vec, i_vec


def apply_extracellular_field(
    sections: list[Any], coords: list[tuple[float, float, float]],
    elec_pos: tuple[float, float, float], current: float,
) -> None:
    """
    Applies the computed extracellular field to each segment of the given sections.
    """
    for sec, (x, y, z) in zip(sections, coords):
        v_ext = point_source_potential(x, y, z, elec_pos, current)
        for seg in sec:
            seg.e_extracellular = v_ext


def run_stimulation_loop(
    sections: list[Any], coords: list[tuple[float, float, float]],
    elec_pos: tuple[float, float, float], i_vec: np.ndarray, dt: float,
    v_init: float = -65.0,
) -> None:
    """
    Runs NEURON stepping loop applying the waveform stepwise.
    """
    from neuron import h

    h.dt = dt
    h.finitialize(v_init)
    for i_t in i_vec:
        apply_extracellular_field(sections, coords, elec_pos, i_t)
        h.fadvance()
