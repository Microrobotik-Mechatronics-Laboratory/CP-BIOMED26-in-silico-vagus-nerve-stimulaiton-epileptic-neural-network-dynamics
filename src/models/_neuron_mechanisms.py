"""
src/models/_neuron_mechanisms.py
==================================
Loads compiled MRG and Sundt C-fiber mechanisms (libnrnmech) into NEURON.
Provides safe relative path handling to support paths with non-ASCII characters or spaces.
"""

import atexit
import glob
import os
import shutil
import tempfile
import warnings
from neuron import h

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MECHANISMS_DIR = os.path.join(_PROJECT_ROOT, "mechanisms")

_loaded = False


def _ascii_safe_path(path: str) -> str:
    """
    Returns an ASCII-safe path that can be loaded by NEURON's HOC interpreter.
    """
    if path.isascii() and " " not in path:
        return path

    try:
        relative = os.path.relpath(path)
    except ValueError:
        relative = None
    if relative is not None and relative.isascii() and " " not in relative:
        return relative

    try:
        link_dir = tempfile.mkdtemp(prefix="nrnmech_")
        atexit.register(shutil.rmtree, link_dir, True)
        link_path = os.path.join(link_dir, os.path.basename(path))
        os.symlink(path, link_path)
        return link_path
    except OSError as exc:
        warnings.warn(
            f"Failed to create symlink for mechanism path: {exc}. Trying raw path.",
            RuntimeWarning,
            stacklevel=3,
        )
        return path


def ensure_mechanisms_loaded() -> None:
    """
    Loads the compiled mechanism library into NEURON once.
    """
    global _loaded
    if _loaded:
        return

    pattern = os.path.join(_MECHANISMS_DIR, "*", "libnrnmech.*")
    candidates = glob.glob(pattern)

    if not candidates:
        # Check parent directory fallback
        parent_pattern = os.path.join(_PROJECT_ROOT, "eski-kod", "mechanisms", "*", "libnrnmech.*")
        candidates = glob.glob(parent_pattern)

    if not candidates:
        warnings.warn(
            f"No compiled mechanism found in {_MECHANISMS_DIR}. NEURON will fallback to default HH.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    lib_path = candidates[0]
    safe_path = _ascii_safe_path(lib_path)
    try:
        h.nrn_load_dll(safe_path)
        _loaded = True
    except Exception as exc:
        warnings.warn(
            f"Failed to load mechanism library from {safe_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
