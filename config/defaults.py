"""
config/defaults.py
==================
Central repository of simulation, biophysical, network, and sweep parameters.
Follows PEP 8 standards and strict elimination of magic numbers.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Electrode and stimulation geometry
# ---------------------------------------------------------------------------
# (x, y, z) coordinates in micrometers.
ELECTRODE_POS: tuple[float, float, float] = (500.0, 0.0, 3000.0)  # µm
DEFAULT_ELEC_DIST: float = 500.0   # µm (x axis distance)
DEFAULT_ELEC_Z: float = 3000.0      # µm (z position along axon)
DEFAULT_TISSUE_RHO: float = 300.0  # Ω·cm (isotropic tissue resistivity)

# ---------------------------------------------------------------------------
# MRG Axon Parameters (McIntyre, Richardson, Grill 2002; ModelDB Acc. 3810)
# Key: fiber diameter (µm)
# Value: (node_diam, node_length, internode_length, n_myelin_lamellae, axon_diam)
# ---------------------------------------------------------------------------
MRG_PARAMS: dict[float, tuple[float, float, int, int, float]] = {
    5.7:  (1.9, 1.0,  500,  80, 3.4),   # Aδ fiber
    8.7:  (2.8, 1.0, 1000, 110, 5.8),   # Aβ fiber (standard target)
    12.8: (4.2, 1.0, 1350, 135, 9.2),   # Aα motor
    16.0: (5.5, 1.0, 1500, 150, 12.7),  # Aα large motor
}

MYELIN_CM_UF_CM2: float = 1.0    # µF/cm² single membrane layer capacitance
MYELIN_G_S_CM2: float = 0.001    # S/cm² single membrane layer leak conductance

DEFAULT_FIBER_DIAM: float = 8.7  # µm (Aβ sensory/tactile fiber)
DEFAULT_N_NODES: int = 15        # Number of Ranvier nodes
DEFAULT_DT: float = 0.005        # ms (5 µs for NEURON cable equation)
DEFAULT_PULSE_WIDTH: float = 0.1 # ms (100 µs pulse width per phase)
DEFAULT_INTERPHASE_GAP: float = 0.05  # ms (gap between biphasic phases)
DEFAULT_V_INIT_MV: float = -65.0 # mV (initial membrane resting potential in NEURON)

# ---------------------------------------------------------------------------
# VNS 30 Parameter Configurations (2 waveforms x 3 amplitudes x 5 frequencies)
# ---------------------------------------------------------------------------
VNS_WAVEFORMS: list[str] = ["dc", "rectangular"]
VNS_AMPLITUDES_MA: list[float] = [1.0, 2.0, 3.0]  # mA
# Unit scaling: 1 mA = 1000 µA for point_source_potential (current in µA)
MA_TO_UA: float = 1000.0

VNS_FREQUENCIES_HZ: list[float] = [1.0, 10.0, 100.0, 1000.0, 10000.0]  # 1 Hz to 10 kHz
DEFAULT_N_REPEATS: int = 5
DEFAULT_BASE_SEED: int = 1000

# ---------------------------------------------------------------------------
# Fiber Bundle & Bridge to Brian2
# ---------------------------------------------------------------------------
DEFAULT_N_FIBERS: int = 20        # Jittered fiber bundle count
DEFAULT_JITTER_SD_MS: float = 0.8 # ms Gaussian jitter standard deviation
AXON_REFRACTORY_MS: float = 1.0   # ms absolute refractory limit

# ---------------------------------------------------------------------------
# Brian2 Simulator Settings & Common LIF Parameters
# ---------------------------------------------------------------------------
DEFAULT_BRIAN_DT_MS: float = 0.1   # ms Brian2 defaultclock.dt
LIF_THRESHOLD_MV: float = -50.0    # mV spike threshold
LIF_RESET_MV: float = -65.0        # mV reset potential
LIF_V_REST_MV: float = -65.0       # mV resting potential
LIF_E_SYN_EX_MV: float = 0.0       # mV AMPA reversal potential
LIF_E_SYN_IN_NORMAL_MV: float = -80.0  # mV GABA healthy hyperpolarizing reversal
LIF_E_SYN_IN_DEP_MV: float = -50.0     # mV GABA depolarizing reversal
LIF_GL_NS: float = 10.0            # nS leak conductance
LIF_TAU_M_EX_MS: float = 20.0      # ms pyramidal membrane time constant
LIF_TAU_M_IN_MS: float = 12.0      # ms interneuron membrane time constant
LIF_REFRACTORY_EX_MS: float = 3.0  # ms excitatory refractory period
LIF_REFRACTORY_IN_MS: float = 1.5  # ms fast-spiking interneuron refractory period

# ---------------------------------------------------------------------------
# NTS Relay Parameters (Level 2)
# Calibrated against Miles (1986), verified against Beaumont et al. (2017)
# ---------------------------------------------------------------------------
NTS_PARAMS: dict[str, float] = {
    "n_neurons": 30,
    "tau_m_ms": 20.0,
    "refractory_ms": 2.0,
    "U": 0.19,
    "tau_f_ms": 20.0,
    "tau_d_ms": 810.0,
    "p": 0.35,
    "w_nS": 12.0,
}

# ---------------------------------------------------------------------------
# Hippocampal Formation Parameters (EC - DG - CA3 - CA1)
# Sclerosis & Sprouting from Aussel et al. (2022); Connectivity from Ahn et al. (2016)
# ---------------------------------------------------------------------------
SUBFIELD_POP_SIZE: dict[str, dict[str, int]] = {
    "EC":  {"ex": 100, "in": 20},
    "DG":  {"ex": 100, "in": 20},
    "CA3": {"ex": 100, "in": 20},
    "CA1": {"ex": 100, "in": 20},
}

# Synaptic Connection Probabilities (Canonical Trisynaptic Loop + Direct Paths)
SYNAPTIC_PROBABILITIES: dict[str, float] = {
    "EC_to_DG_ex": 0.15,
    "EC_to_DG_in": 0.10,
    "EC_to_CA3_ex": 0.08,   # Temporoammonic direct path
    "EC_to_CA1_ex": 0.08,   # Temporoammonic direct path
    "DG_to_CA3_ex": 0.08,   # Mossy fibers
    "DG_to_CA3_in": 0.05,
    "CA3_to_CA3_rec": 0.12, # CA3 recurrent collaterals (11 connections / neuron)
    "CA3_to_CA1_ex": 0.15,  # Schaffer collaterals
    "CA3_to_CA1_in": 0.10,
    "CA1_to_EC_ex": 0.10,   # Subicular feedback loop
    "CA1_to_EC_in": 0.05,
}

# Intra-regional E-I and I-E connection probabilities
INTRA_EI_PROBABILITY: float = 0.25
INTRA_IE_PROBABILITY: float = 0.25

# Synaptic Conductance Weights (nS)
# Tuned for ~480 neuron microcircuit scale where 1 unit represents ~10-50 biological synapses
WEIGHT_AMPA_NS: float = 6.0
WEIGHT_GABA_NS: float = 7.0
WEIGHT_CA3_REC_NS: float = 2.0
WEIGHT_CORTICAL_NOISE_NS: float = 4.5

# ---------------------------------------------------------------------------
# Neuropathology Parameters (Aussel et al. 2022)
# ---------------------------------------------------------------------------
# Sclerosis: fraction of principal cell reduction (scl in [0, 1])
# Active count = N_0 * (1 - coeff * scl)
SCLEROSIS_COEFFS: dict[str, float] = {
    "EC":  0.75,
    "DG":  0.60,
    "CA3": 0.80,
    "CA1": 0.80,
}

# Mossy Fiber Sprouting (spr in [0, 1])
# Recurrent DG granule cell excitation: p_DG_DG = 0.10 * spr
# Sprouted drive to interneurons: p_DG_IN = 0.06 * (1 + spr)
SPROUTING_BASE_PROB: float = 0.10
SPROUTING_IN_BASE_PROB: float = 0.06

# Dynamic Chloride Parameters
CHLORIDE_TAU_HEALTHY_MS: float = 100.0   # Fast extrusion
CHLORIDE_TAU_EPILEPTIC_MS: float = 800.0 # Slow extrusion (impaired KCC2)
CHLORIDE_SPIKE_INCREMENT: float = 0.2     # Delta [Cl] per action potential
CHLORIDE_THRESHOLD: float = 0.5           # Above this, E_GABA shifts to -50 mV

# Default Pathology Presets
PATHOLOGY_PRESETS: dict[str, dict[str, float]] = {
    "healthy": {
        "scl": 0.0,
        "spr": 0.0,
        "tau_cl_ms": CHLORIDE_TAU_HEALTHY_MS,
    },
    "epileptic": {
        "scl": 0.5,
        "spr": 0.8,
        "tau_cl_ms": CHLORIDE_TAU_EPILEPTIC_MS,
    },
}

# Background cortical input Poisson rate driving EC
CORTICAL_POISSON_RATE_HZ: float = 35.0

# VNS Projection to Hippocampus (facilitating TM synapse)
VNS_TO_HIPP_PARAMS: dict[str, float] = {
    "U": 0.2,
    "tau_f_ms": 150.0,
    "tau_d_ms": 300.0,
    "p_ca3": 0.30,
    "p_dg": 0.15,
    "w_nS": 5.0,
}

# ---------------------------------------------------------------------------
# LFP Simulation and Spectral Analysis Parameters
# ---------------------------------------------------------------------------
LFP_SAMPLING_RATE_HZ: float = 1024.0
LFP_FILTER_LOW_HZ: float = 0.15
LFP_FILTER_HIGH_HZ: float = 480.0
THETA_BAND_HZ: tuple[float, float] = (4.0, 8.0)
DEFAULT_SIM_DURATION_MS: float = 1000.0  # 1.0 second standard evaluation window
