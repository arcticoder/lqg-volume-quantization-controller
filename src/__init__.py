#!/usr/bin/env python3
"""
LQG Volume Quantization Controller Package
==========================================

This package provides discrete spacetime V_min patch management using
SU(2) representation control j(j+1) for the LQG FTL Metric Engineering ecosystem.

Core Components:
- VolumeQuantizationController: Main controller for discrete spacetime patches
- SU2RepresentationController: SU(2) quantum number management
- DiscreteSpacetimePatchManager: Patch lifecycle management
- SU2MathematicalIntegrator: Integration with SU(2) mathematical toolkit
- LQGFoundationIntegrator: Integration with LQG framework

Author: LQG Volume Quantization Team
Date: July 5, 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "LQG Volume Quantization Team"
__email__ = "lqg-volume-quantization@example.com"
__license__ = "MIT"

# Physical constants
PLANCK_LENGTH = 1.616e-35  # m
PLANCK_TIME = 5.391e-44    # s
PLANCK_MASS = 2.176e-8     # kg
PLANCK_VOLUME = PLANCK_LENGTH**3
IMMIRZI_GAMMA = 0.2375     # Barbero-Immirzi parameter
HBAR = 1.055e-34          # Reduced Planck constant

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Physical constants
    'PLANCK_LENGTH',
    'PLANCK_TIME', 
    'PLANCK_MASS',
    'PLANCK_VOLUME',
    'IMMIRZI_GAMMA',
    'HBAR',
]

# Import note: Actual classes imported in submodules to avoid circular imports
# Use: from lqg_volume_quantization_controller.core import VolumeQuantizationController
