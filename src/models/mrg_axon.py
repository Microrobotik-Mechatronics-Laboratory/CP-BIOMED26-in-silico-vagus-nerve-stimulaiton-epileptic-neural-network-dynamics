"""
src/models/mrg_axon.py
=======================
McIntyre-Richardson-Grill (2002) myelinated nerve fiber model in NEURON.
ModelDB Accession: 3810.
"""

from typing import Any
from neuron import h

from config.defaults import MRG_PARAMS, MYELIN_CM_UF_CM2, MYELIN_G_S_CM2
from src.models._neuron_mechanisms import ensure_mechanisms_loaded

ensure_mechanisms_loaded()
h.load_file("stdrun.hoc")


def _has_mrg_mechanism() -> bool:
    """Check if compiled MRG axnode mechanism is available in NEURON."""
    ensure_mechanisms_loaded()
    return hasattr(h, "axnode") or hasattr(h, "MRGaxon")


class MRGAxon:
    """
    McIntyre-Richardson-Grill (2002) double-cable myelinated axon model.

    Parameters
    ----------
    diameter_um : float, optional
        Fiber diameter (µm). Defined in MRG_PARAMS: 5.7, 8.7, 12.8, 16.0.
    n_nodes : int, optional
        Number of Ranvier nodes. Default 15.
    """

    def __init__(self, diameter_um: float = 8.7, n_nodes: int = 15) -> None:
        if diameter_um not in MRG_PARAMS:
            raise ValueError(
                f"Unsupported fiber diameter: {diameter_um} µm. "
                f"Valid values: {list(MRG_PARAMS.keys())}"
            )
        self.diameter = diameter_um
        self.n_nodes = n_nodes

        node_diam, node_len, internode_len, n_lamellae, axon_diam = MRG_PARAMS[diameter_um]
        self._axon_diam = axon_diam
        self._node_diam = node_diam
        self._node_len = node_len
        self._internode_len = internode_len
        self._n_lamellae = n_lamellae

        self.nodes: list[Any] = []
        self.internodes: list[Any] = []
        self._nc: Any = None
        self._spike_times: Any = None
        self._build()

    def _build(self) -> None:
        """Constructs axon geometry and ion channels."""
        use_mrg = _has_mrg_mechanism()

        # Nodes of Ranvier
        for i in range(self.n_nodes):
            node = h.Section(name=f"node_{i}")
            node.diam = self._node_diam
            node.L = self._node_len
            node.nseg = 1
            node.insert("extracellular")

            if use_mrg:
                node.insert("axnode")
            else:
                node.insert("hh")
                node.insert("pas")
                node.g_pas = 0.0001
                node.e_pas = -80.0

            self.nodes.append(node)

        # Internodes
        for i in range(self.n_nodes - 1):
            inter = h.Section(name=f"internode_{i}")
            inter.diam = self._axon_diam
            inter.L = self._internode_len
            inter.nseg = 6
            inter.insert("extracellular")
            inter.insert("pas")

            n_layers = 2 * self._n_lamellae
            inter.cm = MYELIN_CM_UF_CM2 / n_layers
            inter.g_pas = MYELIN_G_S_CM2 / n_layers
            inter.e_pas = -80.0

            self.internodes.append(inter)

        # Topological connection
        for i in range(self.n_nodes - 1):
            self.internodes[i].connect(self.nodes[i](1), 0)
            self.nodes[i + 1].connect(self.internodes[i](1), 0)

    def all_sections(self) -> list[Any]:
        """Returns all sections (nodes + internodes) in sequential order."""
        sections = []
        for i in range(self.n_nodes - 1):
            sections.append(self.nodes[i])
            sections.append(self.internodes[i])
        sections.append(self.nodes[-1])
        return sections

    def section_coords(self) -> list[tuple[float, float, float]]:
        """Calculates (x, y, z) midpoint coordinates for each section."""
        coords = []
        z = 0.0
        for i in range(self.n_nodes - 1):
            coords.append((0.0, 0.0, z + self._node_len / 2.0))
            z += self._node_len
            coords.append((0.0, 0.0, z + self._internode_len / 2.0))
            z += self._internode_len
        coords.append((0.0, 0.0, z + self._node_len / 2.0))
        return coords

    def record_last_node_spikes(self) -> Any:
        """Sets up NetCon monitor recording distal spikes from the last node."""
        last_node = self.nodes[-1]
        nc = h.NetCon(last_node(0.5)._ref_v, None, sec=last_node)
        nc.threshold = -20.0  # mV
        spike_times = h.Vector()
        nc.record(spike_times)
        self._nc = nc
        self._spike_times = spike_times
        return spike_times
