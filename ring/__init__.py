"""RING: Range-Indexed Navigation Graph for range-filtered approximate nearest-neighbor search.

The C++ extension ``_frans_v11`` is built into this directory by CMake (see README).
Internal module names (``frans_*``) come from the method's development history; ``FullIndex``
is the RING index used in the paper's experiments.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # the native module is imported as top-level ``_frans_v11``

from .frans_fullscale_v1.model import FullIndex, PARAMETERS  # noqa: E402
from .api import members, prepare, search_window, window  # noqa: E402

__all__ = ["FullIndex", "PARAMETERS", "members", "prepare", "search_window", "window"]
