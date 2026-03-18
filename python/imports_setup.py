"""Venv injection and optional dependency imports for GMM-SLAM.

Exports feature flags:
    HAS_SOGMM, HAS_GMM_REGISTRATION, HAS_GTSAM, HAS_GTSAM_UNSTABLE
and the optional modules themselves when available.
"""

import sys
import os
import glob
import logging as _logging

import numpy as np


def _inject_venv(venv_root: str):
    """Add site-packages from a venv into sys.path."""
    pattern = os.path.join(venv_root, "lib", "python3*", "site-packages")
    for sp in glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)


_GIRA_WS = os.environ.get("GIRA_WS", "/root/gira_ws")
_RECONSTRUCTION_VENV = os.path.join(_GIRA_WS, "gira3d-reconstruction", ".venv")
_REGISTRATION_VENV = os.path.join(_GIRA_WS, "gira3d-registration", ".venv")

_inject_venv(_RECONSTRUCTION_VENV)
_inject_venv(_REGISTRATION_VENV)

# --- SOGMM (reconstruction) ------------------------------------------------
try:
    from sogmm_py.sogmm import SOGMM  # noqa: F401

    HAS_SOGMM = True
except ImportError:
    SOGMM = None
    HAS_SOGMM = False
    _logging.warning(
        f"sogmm_py not found - SOGMM fitting disabled. "
        f"Expected venv at {_RECONSTRUCTION_VENV}"
    )

# --- D2D registration ------------------------------------------------------
try:
    import gmm_d2d_registration_py  # noqa: F401
    from utils.save_gmm import save as save_gmm_official  # noqa: F401

    HAS_GMM_REGISTRATION = True
except ImportError:
    gmm_d2d_registration_py = None
    save_gmm_official = None
    HAS_GMM_REGISTRATION = False
    _logging.warning(
        f"gmm_d2d_registration_py not found - D2D registration disabled. "
        f"Expected venv at {_REGISTRATION_VENV}"
    )

# --- GTSAM ------------------------------------------------------------------
try:
    import gtsam  # noqa: F401
    from gtsam.symbol_shorthand import X  # noqa: F401

    HAS_GTSAM = True
except ImportError:
    gtsam = None
    X = None
    HAS_GTSAM = False
    _logging.warning("gtsam not found - fixed-lag backend disabled.")

try:
    import gtsam_unstable  # noqa: F401

    HAS_GTSAM_UNSTABLE = True
except ImportError:
    gtsam_unstable = None
    HAS_GTSAM_UNSTABLE = False
