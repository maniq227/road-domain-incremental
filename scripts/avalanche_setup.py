#!/usr/bin/env python3
"""
Thin wrapper to expose Avalanche setup utilities under a stable module name.

This allows scripts to import `avalanche_setup` regardless of the original
filename of the implementation module (e.g., `03_avalanche_setup.py`).
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path

_impl_path = Path(__file__).with_name("03_avalanche_setup.py")
_mod = SourceFileLoader("_road_avalanche_setup_impl", str(_impl_path)).load_module()

# Re-export public API
ROADAvalancheSetup = _mod.ROADAvalancheSetup
ROADDataset = _mod.ROADDataset


