"""
src/network/bridge.py
======================
NEURON to Brian2 bridge: converts distal axon spike trains into jittered fiber bundles.
"""

from typing import Iterable, Sequence
import numpy as np
from brian2 import SpikeGeneratorGroup, ms

from config.defaults import AXON_REFRACTORY_MS, DEFAULT_BRIAN_DT_MS, DEFAULT_JITTER_SD_MS


def _build_spike_generator(
    n_fibers: int, all_times_ms: Sequence[float], all_indices: Sequence[int],
) -> SpikeGeneratorGroup:
    """
    Constructs a SpikeGeneratorGroup sorted in time, filtered with axonal refractoriness.
    """
    times_arr = np.asarray(all_times_ms, dtype=float)
    index_arr = np.asarray(all_indices, dtype=np.int64)

    kept_times: list[np.ndarray] = []
    kept_index: list[np.ndarray] = []

    for fiber in range(n_fibers):
        fiber_times = np.sort(times_arr[index_arr == fiber])
        if fiber_times.size == 0:
            continue

        # 1. Enforce absolute refractory period
        accepted: list[float] = []
        last = -np.inf
        for t in fiber_times:
            if t - last >= AXON_REFRACTORY_MS:
                accepted.append(float(t))
                last = float(t)

        # 2. Align to Brian2 time step grid to prevent collisions in single dt bin
        bins = np.unique(np.floor(np.array(accepted) / DEFAULT_BRIAN_DT_MS).astype(np.int64))
        kept_times.append((bins + 0.5) * DEFAULT_BRIAN_DT_MS)
        kept_index.append(np.full(bins.size, fiber, dtype=np.int64))

    if not kept_times:
        return SpikeGeneratorGroup(n_fibers, np.array([], dtype=np.int64), np.array([]) * ms)

    times = np.concatenate(kept_times)
    indices = np.concatenate(kept_index)
    order = np.argsort(times, kind="stable")
    return SpikeGeneratorGroup(n_fibers, indices[order], times[order] * ms)


def bundle_from_single_fiber(
    spike_times_ms: Sequence[float], n_fibers: int = 20,
    jitter_ms: float = DEFAULT_JITTER_SD_MS, seed: int = 0,
) -> SpikeGeneratorGroup:
    """
    Creates a jittered bundle of fibers from a single axon spike train.

    Parameters
    ----------
    spike_times_ms : sequence of float
        Spike times from the distal node in ms.
    n_fibers : int, optional
        Number of fibers in bundle. Default 20.
    jitter_ms : float, optional
        Standard deviation of Gaussian jitter (ms). Default 0.8 ms.
    seed : int, optional
        Random seed for reproducibility. Default 0.

    Returns
    -------
    brian2.SpikeGeneratorGroup
    """
    rng = np.random.default_rng(seed)
    all_times, all_idx = [], []
    for fib in range(n_fibers):
        if len(spike_times_ms) > 0:
            jittered = np.array(spike_times_ms) + rng.normal(0, jitter_ms, size=len(spike_times_ms))
            jittered = np.clip(jittered, 0, None)
            all_times.extend(jittered.tolist())
            all_idx.extend([fib] * len(spike_times_ms))
    return _build_spike_generator(n_fibers, all_times, all_idx)
