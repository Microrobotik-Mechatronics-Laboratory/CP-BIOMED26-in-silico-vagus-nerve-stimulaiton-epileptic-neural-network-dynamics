"""
src/pipeline/level1.py
=======================
Level 1 pipeline: NEURON peripheral MRG axon stimulation and distal spike extraction.
"""

from config.defaults import (
    DEFAULT_DT,
    DEFAULT_PULSE_WIDTH,
    DEFAULT_N_NODES,
    DEFAULT_FIBER_DIAM,
    DEFAULT_ELEC_DIST,
    DEFAULT_ELEC_Z,
    DEFAULT_V_INIT_MV,
    MA_TO_UA,
)
from src.models.mrg_axon import MRGAxon
from src.stim.extracellular_field import biphasic_waveform, run_stimulation_loop


def run_level1(
    freq_hz: float = 20.0,
    amp_ma: float = 1.0,
    duration_ms: float = 200.0,
    dt: float = DEFAULT_DT,
    fiber_diam: float = DEFAULT_FIBER_DIAM,
    n_nodes: int = DEFAULT_N_NODES,
    elec_pos: tuple[float, float, float] = (DEFAULT_ELEC_DIST, 0.0, DEFAULT_ELEC_Z),
    pulse_width_ms: float = DEFAULT_PULSE_WIDTH,
    waveform: str = "rectangular",
) -> list[float]:
    """
    Simulates MRG axon stimulation in NEURON and returns distal spike timestamps.

    Parameters
    ----------
    freq_hz : float
        Stimulation frequency in Hz.
    amp_ma : float
        Current amplitude in milliamperes (mA). Converted to µA internally.
    duration_ms : float
        Simulation duration in ms.
    dt : float, optional
        Time step in ms. Default 0.005 ms.
    fiber_diam : float, optional
        MRG fiber diameter in µm. Default 8.7 µm (Aβ fiber).
    n_nodes : int, optional
        Number of nodes of Ranvier. Default 15.
    elec_pos : tuple of float, optional
        Electrode position in µm. Default (500, 0, 3000).
    pulse_width_ms : float, optional
        Pulse width per phase in ms. Default 0.1 ms.
    waveform : str, optional
        "rectangular" (charge-balanced biphasic) or "dc" (monophasic). Default "rectangular".

    Returns
    -------
    list of float
        Spike arrival times at the distal node in ms.
    """
    fiber = MRGAxon(diameter_um=fiber_diam, n_nodes=n_nodes)
    coords = fiber.section_coords()
    sections = fiber.all_sections()
    spikes = fiber.record_last_node_spikes()

    amp_ua = amp_ma * MA_TO_UA
    t_vec, i_vec = biphasic_waveform(
        freq_hz=freq_hz,
        amp=amp_ua,
        duration_ms=duration_ms,
        dt=dt,
        pulse_width_ms=pulse_width_ms,
        waveform=waveform,
    )

    run_stimulation_loop(sections, coords, elec_pos, i_vec, dt, v_init=DEFAULT_V_INIT_MV)

    return list(spikes)
