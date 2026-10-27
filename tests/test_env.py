"""
tests/test_env.py
=================
Environment and NEURON mechanisms verification test.
"""

import os
import pytest
from neuron import h
import brian2


def test_neuron_and_brian2_installed():
    """Verify Brian2 and NEURON packages are correctly imported."""
    assert brian2.__version__ is not None
    assert hasattr(h, "fadvance")


def test_compiled_mechanisms_loaded():
    """Verify MRG and Sundt mechanisms are present in NEURON."""
    dylib = os.path.join("mechanisms", "arm64", "libnrnmech.dylib")
    assert os.path.exists(dylib), f"Mechanism dylib not found at {dylib}"

    if not hasattr(h, "axnode"):
        h.nrn_load_dll(dylib)

    assert hasattr(h, "axnode"), "MRG mechanism (axnode) failed to load"
    assert hasattr(h, "nahh"), "C-fiber mechanism (nahh) failed to load"
