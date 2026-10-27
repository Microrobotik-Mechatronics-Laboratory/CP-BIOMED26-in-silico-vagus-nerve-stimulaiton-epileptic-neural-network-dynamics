"""
src/network/nts_relay.py
=========================
Brainstem Nucleus Tractus Solitarius (NTS) relay with Tsodyks-Markram short-term plasticity.
Calibrated against Miles (1986), verified against Beaumont et al. (2017).
"""

from typing import Callable
from brian2 import NeuronGroup, Synapses, Quantity, ms, mV, nS

from config.defaults import (
    LIF_THRESHOLD_MV,
    LIF_RESET_MV,
    LIF_V_REST_MV,
    LIF_E_SYN_EX_MV,
    LIF_GL_NS,
    NTS_PARAMS,
)

# Common LIF differential equations for relay populations
NTS_EQS = """
dv/dt = (v_rest - v + ge*(E_syn - v)/gL) / tau_m : volt (unless refractory)
dge/dt = -ge / tau_syn : siemens
v_rest : volt
E_syn  : volt
gL     : siemens
tau_m  : second
tau_syn: second
"""

# Tsodyks-Markram (TM) short-term plasticity equations
TM_SYN_EQS = """
w    : siemens
U    : 1
tau_f: second
tau_d: second
du/dt = -u / tau_f : 1 (event-driven)
dx/dt = (1 - x) / tau_d : 1 (event-driven)
"""

TM_ON_PRE = """
u += U * (1 - u)
ge_post += w * u * x
x -= u * x
"""


def build_nts_relay(
    n_neurons: int = int(NTS_PARAMS["n_neurons"]),
    tau_m: float = NTS_PARAMS["tau_m_ms"],
    refractory_ms: float = NTS_PARAMS["refractory_ms"],
) -> tuple[NeuronGroup, Callable[..., Synapses]]:
    """
    Constructs the NTS relay population and returns factory for afferent TM synapses.

    Parameters
    ----------
    n_neurons : int, optional
        Number of NTS neurons. Default 30.
    tau_m : float, optional
        Membrane time constant in ms. Default 20.0 ms.
    refractory_ms : float, optional
        Refractory period in ms. Default 2.0 ms.

    Returns
    -------
    tuple of (NeuronGroup, Callable)
    """
    nts = NeuronGroup(
        n_neurons, NTS_EQS,
        threshold=f"v > {LIF_THRESHOLD_MV}*mV",
        reset=f"v = {LIF_RESET_MV}*mV",
        refractory=refractory_ms * ms,
        method="euler",
    )
    nts.v = LIF_V_REST_MV * mV
    nts.v_rest = LIF_V_REST_MV * mV
    nts.E_syn = LIF_E_SYN_EX_MV * mV
    nts.gL = LIF_GL_NS * nS
    nts.tau_m = tau_m * ms
    nts.tau_syn = 5.0 * ms

    U_val = NTS_PARAMS["U"]
    tau_f_ms = NTS_PARAMS["tau_f_ms"]
    tau_d_ms = NTS_PARAMS["tau_d_ms"]

    def make_afferent_synapses(
        source: NeuronGroup,
        p: float = NTS_PARAMS["p"],
        w: float = NTS_PARAMS["w_nS"],
    ) -> Synapses:
        """Connects afferent fiber bundle to NTS neurons via depressing TM synapses."""
        syn = Synapses(source, nts, model=TM_SYN_EQS, on_pre=TM_ON_PRE, method="euler")
        syn.connect(p=p)
        syn.w = w * nS
        syn.U = U_val
        syn.tau_f = tau_f_ms * ms
        syn.tau_d = tau_d_ms * ms
        syn.x = 1.0
        syn.u = U_val
        return syn

    return nts, make_afferent_synapses


def make_tm_conn(
    source: NeuronGroup, target: NeuronGroup, U: float, tau_f: Quantity,
    tau_d: Quantity, p: float, w: Quantity, target_var: str = "ge",
) -> Synapses:
    """
    General-purpose Tsodyks-Markram synaptic connector between any two Brian2 groups.
    """
    on_pre = f"""
    u += U * (1 - u)
    {target_var}_post += w * u * x
    x -= u * x
    """
    syn = Synapses(source, target, model=TM_SYN_EQS, on_pre=on_pre, method="euler")
    syn.connect(p=p)
    syn.w = w
    syn.U = U
    syn.tau_f = tau_f
    syn.tau_d = tau_d
    syn.x = 1.0
    syn.u = U
    return syn
