#!/usr/bin/env python3
"""
Core Module for LQG Volume Quantization Controller
==================================================

This module provides the core functionality for discrete spacetime V_min patch
management using SU(2) representation control j(j+1).

Author: LQG Volume Quantization Team
Date: July 5, 2025
Version: 1.0.0
"""

# Import all core classes and functions
from .volume_quantization_controller import (
    VolumeQuantizationController,
    SU2RepresentationController,
    DiscreteSpacetimePatchManager,
    VolumeQuantizationState,
    LQGVolumeConfiguration,
    VolumeQuantizationMode,
    SU2RepresentationScheme,
    create_standard_controller,
    gaussian_volume_distribution,
    planck_scale_volume_distribution
)

from .su2_mathematical_integration import (
    SU2MathematicalIntegrator,
    SU2CalculationResult,
    SU2IntegrationStatus,
    get_su2_integrator,
    compute_3nj_symbol,
    compute_volume_eigenvalue_enhanced,
    compute_representation_matrix
)

from .lqg_foundation_integration import (
    LQGFoundationIntegrator,
    LQGComputationResult,
    LQGIntegrationStatus,
    PolymerQuantizationScheme,
    PolymerQuantizationState,
    get_lqg_integrator,
    compute_polymer_correction,
    compute_lqg_corrected_volume
)

__all__ = [
    # Main controller classes
    'VolumeQuantizationController',
    'SU2RepresentationController', 
    'DiscreteSpacetimePatchManager',
    
    # Data classes
    'VolumeQuantizationState',
    'LQGVolumeConfiguration',
    'SU2CalculationResult',
    'LQGComputationResult',
    'PolymerQuantizationState',
    
    # Enums
    'VolumeQuantizationMode',
    'SU2RepresentationScheme',
    'SU2IntegrationStatus',
    'LQGIntegrationStatus',
    'PolymerQuantizationScheme',
    
    # Integration classes
    'SU2MathematicalIntegrator',
    'LQGFoundationIntegrator',
    
    # Factory functions
    'create_standard_controller',
    'gaussian_volume_distribution',
    'planck_scale_volume_distribution',
    
    # Global integrator functions
    'get_su2_integrator',
    'get_lqg_integrator',
    
    # Convenience functions
    'compute_3nj_symbol',
    'compute_volume_eigenvalue_enhanced',
    'compute_representation_matrix',
    'compute_polymer_correction',
    'compute_lqg_corrected_volume'
]
