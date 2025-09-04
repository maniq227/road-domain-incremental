#!/usr/bin/env python3
"""
Thin wrapper to expose RF-DETR model utilities under a stable module name.

This allows scripts to import `model_setup` regardless of the original
filename of the implementation module (e.g., `02_model_setup.py`).
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path

_impl_path = Path(__file__).with_name("02_model_setup.py")
_mod = SourceFileLoader("_road_model_setup_impl", str(_impl_path)).load_module()

# Re-export public API
RFDETRModel = _mod.RFDETRModel
ROADModelManager = _mod.ROADModelManager


