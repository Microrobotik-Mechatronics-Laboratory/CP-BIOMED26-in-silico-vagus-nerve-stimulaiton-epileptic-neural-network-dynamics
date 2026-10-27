"""
src/network/hippocampus.py
===========================
Epileptic Hippocampal Microcircuit Model (EC - DG - CA3 - CA1).
Integrates neuropathological mechanisms from Aussel et al. (2022) and Ahn et al. (2016):
- Hippocampal Sclerosis (scl)
- Mossy Fiber Sprouting (spr)
- Impaired Chloride Extrusion & Depolarizing GABA (tau_Cl)
- Canonical trisynaptic pathway and intra-regional feedback inhibition
- Upstream VNS modulation from NTS
"""

from typing import Any, Optional
import numpy as np
from brian2 import (
    NeuronGroup,
    Synapses,
    PoissonGroup,
    SpikeMonitor,
    StateMonitor,
    Network,
    ms,
    mV,
    nS,
    Hz,
    defaultclock,
)

from config.defaults import (
    LIF_THRESHOLD_MV,
    LIF_RESET_MV,
    LIF_V_REST_MV,
    LIF_E_SYN_EX_MV,
    LIF_E_SYN_IN_NORMAL_MV,
    LIF_E_SYN_IN_DEP_MV,
    LIF_GL_NS,
    LIF_TAU_M_EX_MS,
    LIF_TAU_M_IN_MS,
    LIF_REFRACTORY_EX_MS,
    LIF_REFRACTORY_IN_MS,
    SUBFIELD_POP_SIZE,
    SYNAPTIC_PROBABILITIES,
    INTRA_EI_PROBABILITY,
    INTRA_IE_PROBABILITY,
    WEIGHT_AMPA_NS,
    WEIGHT_GABA_NS,
    WEIGHT_CA3_REC_NS,
    WEIGHT_CORTICAL_NOISE_NS,
    SCLEROSIS_COEFFS,
    SPROUTING_BASE_PROB,
    SPROUTING_IN_BASE_PROB,
    CHLORIDE_SPIKE_INCREMENT,
    CHLORIDE_THRESHOLD,
    CORTICAL_POISSON_RATE_HZ,
    VNS_TO_HIPP_PARAMS,
)
from src.network.nts_relay import TM_SYN_EQS, TM_ON_PRE, make_tm_conn

# ---------------------------------------------------------------------------
# Differential Equations
# ---------------------------------------------------------------------------

# Excitatory pyramidal/granule neuron with Aussel (2022) dynamic chloride
PYRAMIDAL_EQS = f"""
dv/dt = (v_rest - v + (ge * (E_ex - v) + gi * (E_in_dyn - v)) / gL) / tau_m : volt (unless refractory)
dge/dt = -ge / tau_syn_ex : siemens
dgi/dt = -gi / tau_syn_in : siemens
dCl/dt = -Cl / tau_Cl : 1

E_in_dyn = -80.0*mV + 30.0*mV / (1.0 + exp(-50.0 * (Cl - {CHLORIDE_THRESHOLD}))) : volt
v_rest     : volt
E_ex       : volt
gL         : siemens
tau_m      : second
tau_syn_ex : second
tau_syn_in : second
tau_Cl     : second
"""

PYRAMIDAL_RESET = f"""
v = {LIF_RESET_MV}*mV
Cl = Cl + {CHLORIDE_SPIKE_INCREMENT}
"""

# GABAergic fast-spiking interneuron equations
INTERNEURON_EQS = f"""
dv/dt = (v_rest - v + (ge * (E_ex - v) + gi * (E_in - v)) / gL) / tau_m : volt (unless refractory)
dge/dt = -ge / tau_syn_ex : siemens
dgi/dt = -gi / tau_syn_in : siemens

v_rest     : volt
E_ex       : volt
E_in       : volt
gL         : siemens
tau_m      : second
tau_syn_ex : second
tau_syn_in : second
"""


class HippocampalFormation:
    """
    Complete human hippocampal formation (EC, DG, CA3, CA1) with pathological parameters.
    """

    def __init__(
        self,
        scl: float = 0.0,
        spr: float = 0.0,
        tau_cl_ms: float = 100.0,
        cortical_rate_hz: float = CORTICAL_POISSON_RATE_HZ,
        seed: int = 1000,
    ) -> None:
        self.scl = float(np.clip(scl, 0.0, 1.0))
        self.spr = float(np.clip(spr, 0.0, 1.0))
        self.tau_cl_ms = float(tau_cl_ms)
        self.cortical_rate_hz = float(cortical_rate_hz)
        self.seed = seed

        self.populations: dict[str, dict[str, NeuronGroup]] = {}
        self.synapses: list[Synapses] = []
        self.monitors: dict[str, Any] = {}
        self.cortical_input: Optional[PoissonGroup] = None
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Constructs populations, path connections, and neuropathologies."""
        # 1. Instantiate Excitatory and Inhibitory populations for each region
        for region in ["EC", "DG", "CA3", "CA1"]:
            base_ex = SUBFIELD_POP_SIZE[region]["ex"]
            n_in = SUBFIELD_POP_SIZE[region]["in"]

            # Hippocampal Sclerosis (Aussel et al. 2022)
            coeff = SCLEROSIS_COEFFS[region]
            n_ex = max(5, int(np.round(base_ex * (1.0 - coeff * self.scl))))

            # Excitatory group
            grp_ex = NeuronGroup(
                n_ex,
                PYRAMIDAL_EQS,
                threshold=f"v > {LIF_THRESHOLD_MV}*mV",
                reset=PYRAMIDAL_RESET,
                refractory=LIF_REFRACTORY_EX_MS * ms,
                method="euler",
            )
            grp_ex.v = LIF_V_REST_MV * mV
            grp_ex.v_rest = LIF_V_REST_MV * mV
            grp_ex.E_ex = LIF_E_SYN_EX_MV * mV
            grp_ex.gL = LIF_GL_NS * nS
            grp_ex.tau_m = LIF_TAU_M_EX_MS * ms
            grp_ex.tau_syn_ex = 3.0 * ms
            grp_ex.tau_syn_in = 8.0 * ms
            grp_ex.tau_Cl = self.tau_cl_ms * ms
            grp_ex.Cl = 0.0

            # Inhibitory group (spared in sclerosis)
            grp_in = NeuronGroup(
                n_in,
                INTERNEURON_EQS,
                threshold=f"v > {LIF_THRESHOLD_MV}*mV",
                reset=f"v = {LIF_RESET_MV}*mV",
                refractory=LIF_REFRACTORY_IN_MS * ms,
                method="euler",
            )
            grp_in.v = LIF_V_REST_MV * mV
            grp_in.v_rest = LIF_V_REST_MV * mV
            grp_in.E_ex = LIF_E_SYN_EX_MV * mV
            grp_in.E_in = LIF_E_SYN_IN_NORMAL_MV * mV
            grp_in.gL = LIF_GL_NS * nS
            grp_in.tau_m = LIF_TAU_M_IN_MS * ms
            grp_in.tau_syn_ex = 3.0 * ms
            grp_in.tau_syn_in = 8.0 * ms

            self.populations[region] = {"ex": grp_ex, "in": grp_in}

        # Helper connectors
        def _connect_ampa(src: NeuronGroup, dst: NeuronGroup, p: float, w_ns: float) -> Synapses:
            syn = Synapses(src, dst, on_pre="ge_post += w", model="w : siemens")
            syn.connect(p=p)
            syn.w = w_ns * nS
            self.synapses.append(syn)
            return syn

        def _connect_gaba(src: NeuronGroup, dst: NeuronGroup, p: float, w_ns: float) -> Synapses:
            syn = Synapses(src, dst, on_pre="gi_post += w", model="w : siemens")
            syn.connect(p=p)
            syn.w = w_ns * nS
            self.synapses.append(syn)
            return syn

        # 2. Intra-regional Feedback & Feedforward Inhibition
        for region in ["EC", "DG", "CA3", "CA1"]:
            pex = self.populations[region]["ex"]
            pin = self.populations[region]["in"]
            _connect_ampa(pex, pin, p=INTRA_EI_PROBABILITY, w_ns=WEIGHT_AMPA_NS)
            _connect_gaba(pin, pex, p=INTRA_IE_PROBABILITY, w_ns=WEIGHT_GABA_NS)
            _connect_gaba(pin, pin, p=0.15, w_ns=WEIGHT_GABA_NS)

        # 3. Canonical Pathways
        ec_ex = self.populations["EC"]["ex"]
        dg_ex = self.populations["DG"]["ex"]
        dg_in = self.populations["DG"]["in"]
        ca3_ex = self.populations["CA3"]["ex"]
        ca3_in = self.populations["CA3"]["in"]
        ca1_ex = self.populations["CA1"]["ex"]
        ca1_in = self.populations["CA1"]["in"]
        ec_in = self.populations["EC"]["in"]

        # Perforant Path: EC -> DG & temporoammonic paths
        _connect_ampa(ec_ex, dg_ex, p=SYNAPTIC_PROBABILITIES["EC_to_DG_ex"], w_ns=WEIGHT_AMPA_NS)
        _connect_ampa(ec_ex, dg_in, p=SYNAPTIC_PROBABILITIES["EC_to_DG_in"], w_ns=WEIGHT_AMPA_NS)
        _connect_ampa(ec_ex, ca3_ex, p=SYNAPTIC_PROBABILITIES["EC_to_CA3_ex"], w_ns=WEIGHT_AMPA_NS)
        _connect_ampa(ec_ex, ca1_ex, p=SYNAPTIC_PROBABILITIES["EC_to_CA1_ex"], w_ns=WEIGHT_AMPA_NS)

        # Mossy Fibers: DG -> CA3
        _connect_ampa(dg_ex, ca3_ex, p=SYNAPTIC_PROBABILITIES["DG_to_CA3_ex"], w_ns=WEIGHT_AMPA_NS * 1.5)
        _connect_ampa(dg_ex, ca3_in, p=SYNAPTIC_PROBABILITIES["DG_to_CA3_in"], w_ns=WEIGHT_AMPA_NS)

        # CA3 Recurrent Collaterals
        rec_syn = Synapses(ca3_ex, ca3_ex, on_pre="ge_post += w", model="w : siemens")
        rec_syn.connect(condition="i != j", p=SYNAPTIC_PROBABILITIES["CA3_to_CA3_rec"])
        rec_syn.w = WEIGHT_CA3_REC_NS * nS
        self.synapses.append(rec_syn)

        # Schaffer Collaterals: CA3 -> CA1
        _connect_ampa(ca3_ex, ca1_ex, p=SYNAPTIC_PROBABILITIES["CA3_to_CA1_ex"], w_ns=WEIGHT_AMPA_NS)
        _connect_ampa(ca3_ex, ca1_in, p=SYNAPTIC_PROBABILITIES["CA3_to_CA1_in"], w_ns=WEIGHT_AMPA_NS)

        # Subicular Feedback: CA1 -> EC
        _connect_ampa(ca1_ex, ec_ex, p=SYNAPTIC_PROBABILITIES["CA1_to_EC_ex"], w_ns=WEIGHT_AMPA_NS)
        _connect_ampa(ca1_ex, ec_in, p=SYNAPTIC_PROBABILITIES["CA1_to_EC_in"], w_ns=WEIGHT_AMPA_NS)

        # 4. Mossy Fiber Sprouting (Aussel et al. 2022)
        if self.spr > 0.0:
            p_spr_ex = SPROUTING_BASE_PROB * self.spr
            sprout_syn = Synapses(dg_ex, dg_ex, on_pre="ge_post += w", model="w : siemens")
            sprout_syn.connect(condition="i != j", p=p_spr_ex)
            sprout_syn.w = WEIGHT_AMPA_NS * nS
            self.synapses.append(sprout_syn)

            p_spr_in = SPROUTING_IN_BASE_PROB * (1.0 + self.spr)
            _connect_ampa(dg_ex, dg_in, p=p_spr_in, w_ns=WEIGHT_AMPA_NS)

        # 5. Background Cortical Noise to EC
        if self.cortical_rate_hz > 0.0:
            self.cortical_input = PoissonGroup(30, self.cortical_rate_hz * Hz)
            _connect_ampa(self.cortical_input, ec_ex, p=0.20, w_ns=WEIGHT_CORTICAL_NOISE_NS)
            _connect_ampa(self.cortical_input, ec_in, p=0.15, w_ns=WEIGHT_CORTICAL_NOISE_NS)

    def attach_vns_input(self, nts_relay_group: Any) -> list[Synapses]:
        """
        Connects NTS relay to CA3 and Perforant Path via facilitating TM synapses.
        """
        vns_synapses = []
        ca3_ex = self.populations["CA3"]["ex"]
        ca3_in = self.populations["CA3"]["in"]
        dg_ex = self.populations["DG"]["ex"]
        ec_ex = self.populations["EC"]["ex"]

        u_val = VNS_TO_HIPP_PARAMS["U"]
        tf = VNS_TO_HIPP_PARAMS["tau_f_ms"] * ms
        td = VNS_TO_HIPP_PARAMS["tau_d_ms"] * ms
        w = VNS_TO_HIPP_PARAMS["w_nS"] * nS

        # TM facilitating synapses to CA3
        syn_ca3_ex = make_tm_conn(nts_relay_group, ca3_ex, U=u_val, tau_f=tf, tau_d=td,
                                  p=VNS_TO_HIPP_PARAMS["p_ca3"], w=w)
        syn_ca3_in = make_tm_conn(nts_relay_group, ca3_in, U=u_val, tau_f=tf, tau_d=td,
                                  p=0.20, w=w)
        # TM synapse to Perforant Path targets
        syn_dg_ex = make_tm_conn(nts_relay_group, dg_ex, U=u_val, tau_f=tf, tau_d=td,
                                 p=VNS_TO_HIPP_PARAMS["p_dg"], w=w)
        syn_ec_ex = make_tm_conn(nts_relay_group, ec_ex, U=u_val, tau_f=tf, tau_d=td,
                                 p=0.10, w=w)

        vns_synapses.extend([syn_ca3_ex, syn_ca3_in, syn_dg_ex, syn_ec_ex])
        self.synapses.extend(vns_synapses)
        return vns_synapses

    def setup_monitors(self, record_lfp: bool = False) -> dict[str, Any]:
        """Sets up SpikeMonitors for all subfields, and optional StateMonitors for LFP."""
        self.monitors = {}
        for region in ["EC", "DG", "CA3", "CA1"]:
            self.monitors[f"spike_{region}_ex"] = SpikeMonitor(self.populations[region]["ex"])
            self.monitors[f"spike_{region}_in"] = SpikeMonitor(self.populations[region]["in"])
            if record_lfp:
                # State monitor for LFP synaptic current reconstruction and chloride dynamics
                self.monitors[f"state_{region}_ex"] = StateMonitor(
                    self.populations[region]["ex"],
                    variables=["v", "ge", "gi", "Cl"],
                    record=True,
                    dt=1.0 * ms,
                )
        return self.monitors

    def all_network_objects(self) -> list[Any]:
        """Returns all Brian2 objects to be added into a Network instance."""
        objs = []
        for reg_dict in self.populations.values():
            objs.extend(reg_dict.values())
        objs.extend(self.synapses)
        if self.cortical_input is not None:
            objs.append(self.cortical_input)
        objs.extend(self.monitors.values())
        return objs
