"""
tachorpm - Extract rotational speed from tachometer pulse signals.

A Python implementation of MATLAB's tachorpm function for vibration analysis.
"""

from .tachorpm import (
    TachoResult,
    statelevels,
    tachorpm,
    tachorpm_simple,
)

__version__ = "0.1.0"
__all__ = ["tachorpm", "tachorpm_simple", "TachoResult", "statelevels"]
