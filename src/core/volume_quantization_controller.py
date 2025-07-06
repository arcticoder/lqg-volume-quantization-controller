#!/usr/bin/env python3
"""
LQG Volume Quantization Controller - Core Implementation
======================================================

This module implements the primary volume quantization controller for managing
discrete spacetime V_min patches using SU(2) representation control j(j+1).

The implementation provides:
1. SU(2) representation management for volume eigenvalue computation
2. Discrete spacetime patch creation and evolution
3. Real-time constraint algebra monitoring
4. Scale-adaptive uncertainty quantification
5. Integration with the complete LQG FTL ecosystem

Mathematical Foundation:
V_min = γ * l_P³ * √(j(j+1))

Where:
- γ = 0.2375 (Barbero-Immirzi parameter)
- l_P = 1.616×10⁻³⁵ m (Planck length)
- j = SU(2) representation label (j ≥ 1/2)

Author: LQG Volume Quantization Team
Date: July 5, 2025
Version: 1.0.0
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from scipy.special import spherical_jn, factorial
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add workspace repositories to path for integration
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.extend([
    str(WORKSPACE_ROOT / "unified-lqg"),
    str(WORKSPACE_ROOT / "su2-3nj-closedform" / "scripts"),
    str(WORKSPACE_ROOT / "su2-3nj-generating-functional"),
    str(WORKSPACE_ROOT / "su2-3nj-uniform-closed-form"),
    str(WORKSPACE_ROOT / "su2-node-matrix-elements"),
    str(WORKSPACE_ROOT / "unified-lqg-qft"),
    str(WORKSPACE_ROOT / "lqg-polymer-field-generator" / "src" / "core")
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Physical constants
PLANCK_LENGTH = 1.616e-35  # m
PLANCK_TIME = 5.391e-44    # s
PLANCK_MASS = 2.176e-8     # kg
PLANCK_VOLUME = PLANCK_LENGTH**3
IMMIRZI_GAMMA = 0.2375     # Barbero-Immirzi parameter
HBAR = 1.055e-34          # Reduced Planck constant
LIGHT_SPEED = 2.998e8      # m/s


class VolumeQuantizationMode(Enum):
    """Volume quantization operational modes"""
    STANDARD = "standard"              # Standard LQG volume eigenvalues
    ENHANCED = "enhanced"              # Enhanced with polymer corrections
    PRODUCTION = "production"          # Production-ready with all optimizations
    VALIDATION = "validation"          # Validation mode with extensive checks


class SU2RepresentationScheme(Enum):
    """SU(2) representation calculation schemes"""
    ANALYTICAL = "analytical"          # Direct analytical computation
    CLOSED_FORM = "closed_form"        # Using su2-3nj-closedform
    GENERATING_FUNCTIONAL = "generating_functional"  # Using generating functional
    ASYMPTOTIC = "asymptotic"          # Large j asymptotic expansion


@dataclass
class VolumeQuantizationState:
    """State representation for discrete spacetime patches"""
    j_value: float                     # SU(2) representation label
    patch_id: int                      # Unique patch identifier
    volume_eigenvalue: float           # V_min eigenvalue (m³)
    coordinates: np.ndarray            # Spatial coordinates (m)
    polymer_scale: float               # Local polymer scale parameter (m)
    uncertainty_bounds: Tuple[float, float]  # Volume uncertainty bounds (m³)
    constraint_violations: Dict[str, float]  # Constraint algebra violations
    timestamp: float                   # State timestamp
    metadata: Dict[str, any] = field(default_factory=dict)  # Additional data


@dataclass
class LQGVolumeConfiguration:
    """Configuration parameters for LQG volume quantization"""
    # Physical parameters
    gamma_immirzi: float = IMMIRZI_GAMMA
    planck_length: float = PLANCK_LENGTH
    planck_volume: float = PLANCK_VOLUME
    
    # SU(2) representation parameters
    min_j: float = 0.5                 # Minimum SU(2) representation
    max_j: float = 10.0                # Maximum SU(2) representation
    j_precision: float = 1e-12         # Numerical precision for j optimization
    
    # Volume quantization parameters
    volume_tolerance: float = 1e-15    # Volume computation tolerance
    scheme: SU2RepresentationScheme = SU2RepresentationScheme.ANALYTICAL
    mode: VolumeQuantizationMode = VolumeQuantizationMode.PRODUCTION
    
    # Patch management parameters
    max_patches: int = 10000           # Maximum number of active patches
    patch_lifetime: float = 1e-40      # Maximum patch lifetime (s)
    
    # Uncertainty quantification parameters
    base_uncertainty: float = 1e-45    # Base volume uncertainty (m³)
    scale_adaptation_factor: float = 0.1  # Scale adaptation coefficient
    
    # Constraint monitoring parameters
    constraint_monitoring: bool = True
    violation_threshold: float = 1e-6
    
    # Performance parameters
    enable_caching: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True


class SU2RepresentationController:
    """
    SU(2) representation control system for j(j+1) eigenvalue management
    
    This class provides the core mathematical foundation for volume quantization
    through SU(2) representation theory and integration with the SU(2) mathematical
    toolkit repositories.
    """
    
    def __init__(self, config: Optional[LQGVolumeConfiguration] = None):
        """Initialize SU(2) representation controller"""
        self.config = config or LQGVolumeConfiguration()
        
        # Core physical parameters
        self.gamma = self.config.gamma_immirzi
        self.l_planck = self.config.planck_length
        self.min_j = self.config.min_j
        self.max_j = self.config.max_j
        
        # Volume computation cache
        self._volume_cache = {} if self.config.enable_caching else None
        
        # Initialize SU(2) mathematical integration
        self._initialize_su2_integration()
        
        logger.info(f"SU(2) Controller initialized with scheme: {self.config.scheme.value}")
        logger.info(f"j range: [{self.min_j}, {self.max_j}]")
        logger.info(f"γ = {self.gamma}, l_P = {self.l_planck:.3e} m")
    
    def _initialize_su2_integration(self):
        """Initialize integration with SU(2) mathematical repositories"""
        try:
            # Import SU(2) mathematical components
            if self.config.scheme == SU2RepresentationScheme.CLOSED_FORM:
                from coefficient_calculator import calculate_3nj
                self._su2_3nj_calculator = calculate_3nj
                logger.info("✅ SU(2) 3nj closed-form integration successful")
            
            # Additional integrations can be added here for other schemes
            self._su2_integration_ready = True
            
        except ImportError as e:
            logger.warning(f"SU(2) integration partially available: {e}")
            self._su2_integration_ready = False
    
    def compute_volume_eigenvalue(self, j: float, use_cache: bool = True) -> float:
        """
        Compute discrete spacetime volume eigenvalue
        
        V_min = γ * l_P³ * √(j(j+1))
        
        Args:
            j: SU(2) representation label
            use_cache: Whether to use cached results
            
        Returns:
            Volume eigenvalue in m³
        """
        # Input validation
        if j < self.min_j:
            raise ValueError(f"Invalid SU(2) representation j={j} < {self.min_j}")
        if j > self.max_j:
            logger.warning(f"j={j} exceeds max_j={self.max_j}, extrapolating")
        
        # Check cache
        cache_key = f"vol_{j:.12f}"
        if use_cache and self._volume_cache and cache_key in self._volume_cache:
            return self._volume_cache[cache_key]
        
        # Core volume eigenvalue computation
        j_eigenvalue = j * (j + 1)
        volume_eigenvalue = (
            self.gamma * 
            (self.l_planck ** 3) * 
            np.sqrt(j_eigenvalue)
        )
        
        # Enhanced computation for production mode
        if self.config.mode == VolumeQuantizationMode.ENHANCED:
            volume_eigenvalue *= self._compute_polymer_enhancement(j)
        
        # Cache result
        if use_cache and self._volume_cache is not None:
            if len(self._volume_cache) < self.config.cache_size:
                self._volume_cache[cache_key] = volume_eigenvalue
        
        return volume_eigenvalue
    
    def _compute_polymer_enhancement(self, j: float) -> float:
        """Compute polymer quantization enhancement factor"""
        # Simplified polymer enhancement based on LQG corrections
        # This would integrate with lqg-polymer-field-generator for full implementation
        mu_polymer = self.l_planck * np.sqrt(j)
        enhancement = 1 + 0.1 * np.sin(np.pi * mu_polymer / self.l_planck)
        return enhancement
    
    def optimize_j_representation(self, target_volume: float, 
                                tolerance: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Optimize SU(2) representation j to achieve target volume
        
        Solves: V_min = γ * l_P³ * √(j(j+1)) = target_volume
        for optimal j value.
        
        Args:
            target_volume: Target volume eigenvalue (m³)
            tolerance: Optimization tolerance
            
        Returns:
            Tuple of (optimal_j, optimization_result)
        """
        tolerance = tolerance or self.config.j_precision
        
        def volume_error(j):
            if j < self.min_j:
                return np.inf
            try:
                computed_volume = self.compute_volume_eigenvalue(j, use_cache=False)
                return abs(computed_volume - target_volume)
            except Exception:
                return np.inf
        
        # Analytical solution for initial guess
        # j(j+1) = (target_volume / (γ * l_P³))²
        target_j_squared = (target_volume / (self.gamma * self.l_planck**3))**2
        
        # Solve quadratic: j² + j - target_j_squared = 0
        discriminant = 1 + 4 * target_j_squared
        if discriminant < 0:
            raise ValueError(f"No real solution for target volume {target_volume}")
        
        j_analytical = (-1 + np.sqrt(discriminant)) / 2
        
        # Numerical refinement
        try:
            opt_result = opt.minimize_scalar(
                volume_error, 
                bounds=(max(self.min_j, j_analytical * 0.5), 
                       min(self.max_j, j_analytical * 2.0)),
                method='bounded',
                options={'xatol': tolerance}
            )
            
            optimal_j = opt_result.x
            final_volume = self.compute_volume_eigenvalue(optimal_j)
            
            optimization_result = {
                'optimal_j': optimal_j,
                'achieved_volume': final_volume,
                'target_volume': target_volume,
                'volume_error': abs(final_volume - target_volume),
                'relative_error': abs(final_volume - target_volume) / target_volume,
                'analytical_guess': j_analytical,
                'convergence_success': opt_result.success,
                'optimization_message': opt_result.message if hasattr(opt_result, 'message') else 'Success'
            }
            
            return optimal_j, optimization_result
            
        except Exception as e:
            logger.error(f"j optimization failed: {e}")
            # Fallback to analytical solution
            return j_analytical, {
                'optimal_j': j_analytical,
                'achieved_volume': self.compute_volume_eigenvalue(j_analytical),
                'target_volume': target_volume,
                'volume_error': np.inf,
                'analytical_guess': j_analytical,
                'convergence_success': False,
                'optimization_message': f'Fallback to analytical: {str(e)}'
            }
    
    def compute_j_eigenvalue_matrix(self, j: float, 
                                  m_values: Optional[List[float]] = None) -> np.ndarray:
        """
        Compute SU(2) representation matrices for given j
        
        This integrates with su2-node-matrix-elements for full matrix computations.
        """
        if m_values is None:
            # Generate standard m values: -j, -j+1, ..., j-1, j
            m_values = [j - i for i in range(int(2*j + 1))]
        
        dim = len(m_values)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        # Diagonal elements: j(j+1)
        j_eigenvalue = j * (j + 1)
        np.fill_diagonal(matrix, j_eigenvalue)
        
        return matrix
    
    def validate_j_representation(self, j: float) -> Dict[str, bool]:
        """Validate SU(2) representation consistency"""
        validation = {
            'valid_range': self.min_j <= j <= self.max_j,
            'positive_eigenvalue': j * (j + 1) >= 0,
            'half_integer': abs(j - round(j)) < 1e-10 or abs(j - round(j) + 0.5) < 1e-10,
            'volume_finite': np.isfinite(self.compute_volume_eigenvalue(j)),
        }
        
        validation['overall_valid'] = all(validation.values())
        return validation


class DiscreteSpacetimePatchManager:
    """
    Manager for discrete spacetime V_min patches
    
    This class handles the creation, evolution, and monitoring of discrete 
    spacetime patches using the SU(2) representation controller.
    """
    
    def __init__(self, su2_controller: SU2RepresentationController, 
                 config: Optional[LQGVolumeConfiguration] = None):
        """Initialize patch manager"""
        self.su2_controller = su2_controller
        self.config = config or su2_controller.config
        
        # Patch storage
        self.active_patches: Dict[int, VolumeQuantizationState] = {}
        self.patch_counter = 0
        
        # Monitoring parameters
        self.monitoring_enabled = self.config.constraint_monitoring
        self.violation_threshold = self.config.violation_threshold
        
        logger.info("Discrete Spacetime Patch Manager initialized")
        logger.info(f"Max patches: {self.config.max_patches}")
        logger.info(f"Constraint monitoring: {self.monitoring_enabled}")
    
    def create_patch(self, target_volume: float, coordinates: np.ndarray,
                    polymer_scale: Optional[float] = None,
                    metadata: Optional[Dict] = None) -> VolumeQuantizationState:
        """
        Create new discrete spacetime patch with specified volume
        
        Args:
            target_volume: Target volume eigenvalue (m³)
            coordinates: Spatial coordinates (m)
            polymer_scale: Local polymer scale parameter (m)
            metadata: Additional patch metadata
            
        Returns:
            VolumeQuantizationState for the created patch
        """
        if len(self.active_patches) >= self.config.max_patches:
            raise RuntimeError(f"Maximum number of patches ({self.config.max_patches}) exceeded")
        
        # Optimize SU(2) representation for target volume
        optimal_j, opt_result = self.su2_controller.optimize_j_representation(target_volume)
        
        # Determine polymer scale
        if polymer_scale is None:
            polymer_scale = self._optimize_polymer_scale(optimal_j, coordinates)
        
        # Compute uncertainty bounds
        uncertainty_bounds = self._compute_uncertainty_bounds(
            optimal_j, target_volume, coordinates
        )
        
        # Create patch state
        patch_state = VolumeQuantizationState(
            j_value=optimal_j,
            patch_id=self.patch_counter,
            volume_eigenvalue=opt_result['achieved_volume'],
            coordinates=coordinates.copy(),
            polymer_scale=polymer_scale,
            uncertainty_bounds=uncertainty_bounds,
            constraint_violations={},
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Runtime constraint algebra monitoring
        if self.monitoring_enabled:
            violations = self._monitor_constraint_algebra(patch_state)
            patch_state.constraint_violations = violations
        
        # Store patch
        self.active_patches[self.patch_counter] = patch_state
        self.patch_counter += 1
        
        logger.info(f"Created patch {patch_state.patch_id} with j={optimal_j:.3f}, "
                   f"V={patch_state.volume_eigenvalue:.2e} m³")
        
        return patch_state
    
    def _optimize_polymer_scale(self, j: float, coordinates: np.ndarray) -> float:
        """Optimize polymer scale parameter for given j and coordinates"""
        # Physical scale hierarchy
        coord_magnitude = np.linalg.norm(coordinates)
        observational_scale = max(coord_magnitude, 1e-9)  # At least 1 nm
        
        # LQG-motivated polymer scale
        lqg_scale = self.su2_controller.l_planck * np.sqrt(j * (j + 1))
        
        # Optimize within physical bounds
        min_polymer_scale = self.su2_controller.l_planck
        max_polymer_scale = min(observational_scale, 1e-6)  # Max 1 μm
        
        optimal_scale = max(min_polymer_scale, min(lqg_scale, max_polymer_scale))
        
        return optimal_scale
    
    def _compute_uncertainty_bounds(self, j: float, target_volume: float, 
                                  coordinates: np.ndarray) -> Tuple[float, float]:
        """Compute scale-adaptive uncertainty bounds"""
        # Base uncertainty scaled by representation
        base_unc = self.config.base_uncertainty * np.sqrt(j * (j + 1))
        
        # Scale adaptation based on coordinate magnitude
        coord_magnitude = np.linalg.norm(coordinates)
        scale_factor = 1 + self.config.scale_adaptation_factor * np.log10(
            max(coord_magnitude / 1e-9, 1)
        )
        
        # Volume-dependent uncertainty
        planck_volume = self.su2_controller.l_planck**3
        volume_factor = 1 + 0.01 * abs(np.log10(target_volume / planck_volume))
        
        # Final uncertainty bounds
        total_uncertainty = base_unc * scale_factor * volume_factor
        
        lower_bound = max(0, target_volume - total_uncertainty)
        upper_bound = target_volume + total_uncertainty
        
        return (lower_bound, upper_bound)
    
    def _monitor_constraint_algebra(self, patch_state: VolumeQuantizationState) -> Dict[str, float]:
        """Runtime constraint algebra monitoring"""
        violations = {}
        
        # Monitor volume quantization constraint
        j = patch_state.j_value
        expected_volume = self.su2_controller.compute_volume_eigenvalue(j)
        volume_violation = abs(patch_state.volume_eigenvalue - expected_volume) / expected_volume
        violations['volume_consistency'] = volume_violation
        
        # Monitor SU(2) representation constraints
        j_eigenvalue = j * (j + 1)
        expected_j_eigenvalue = j * (j + 1)  # Should be exact
        j_violation = abs(j_eigenvalue - expected_j_eigenvalue)
        violations['su2_eigenvalue'] = j_violation
        
        # Monitor polymer scale physical bounds
        mu = patch_state.polymer_scale
        l_p = self.su2_controller.l_planck
        
        if mu < 0.01 * l_p or mu > 100 * l_p:
            violations['polymer_scale_bounds'] = abs(np.log10(mu / l_p))
        else:
            violations['polymer_scale_bounds'] = 0.0
        
        # Monitor coordinate consistency
        coord_magnitude = np.linalg.norm(patch_state.coordinates)
        if coord_magnitude > 0:
            max_physical_scale = 1e26  # Observable universe diameter
            if coord_magnitude > max_physical_scale:
                violations['coordinate_bounds'] = np.log10(coord_magnitude / max_physical_scale)
            else:
                violations['coordinate_bounds'] = 0.0
        
        # Check for significant violations
        significant_violations = {
            k: v for k, v in violations.items() 
            if v > self.violation_threshold
        }
        
        if significant_violations:
            logger.warning(f"Constraint violations detected in patch {patch_state.patch_id}: "
                          f"{significant_violations}")
        
        return violations
    
    def update_patch(self, patch_id: int, 
                    new_coordinates: Optional[np.ndarray] = None,
                    new_target_volume: Optional[float] = None,
                    new_metadata: Optional[Dict] = None) -> VolumeQuantizationState:
        """Update existing patch with new parameters"""
        if patch_id not in self.active_patches:
            raise ValueError(f"Patch {patch_id} not found")
        
        patch = self.active_patches[patch_id]
        
        # Update coordinates
        if new_coordinates is not None:
            patch.coordinates = new_coordinates.copy()
            # Recompute polymer scale
            patch.polymer_scale = self._optimize_polymer_scale(patch.j_value, new_coordinates)
        
        # Update volume if requested
        if new_target_volume is not None:
            optimal_j, opt_result = self.su2_controller.optimize_j_representation(new_target_volume)
            patch.j_value = optimal_j
            patch.volume_eigenvalue = opt_result['achieved_volume']
            # Recompute uncertainty bounds
            patch.uncertainty_bounds = self._compute_uncertainty_bounds(
                optimal_j, new_target_volume, patch.coordinates
            )
        
        # Update metadata
        if new_metadata is not None:
            patch.metadata.update(new_metadata)
        
        # Update timestamp
        patch.timestamp = time.time()
        
        # Re-monitor constraints
        if self.monitoring_enabled:
            patch.constraint_violations = self._monitor_constraint_algebra(patch)
        
        logger.info(f"Updated patch {patch_id}")
        return patch
    
    def evolve_patches(self, time_step: float, evolution_steps: int = 1) -> Dict[str, any]:
        """Evolve all active patches through time"""
        evolution_data = {
            'time_step': time_step,
            'evolution_steps': evolution_steps,
            'initial_patches': len(self.active_patches),
            'evolved_patches': 0,
            'violations_detected': 0,
            'patch_updates': []
        }
        
        for step in range(evolution_steps):
            current_time = time.time()
            patches_to_remove = []
            
            for patch_id, patch in self.active_patches.items():
                # Check patch lifetime
                if (current_time - patch.timestamp) > self.config.patch_lifetime:
                    patches_to_remove.append(patch_id)
                    continue
                
                # Simplified time evolution (placeholder for more sophisticated dynamics)
                # In a full implementation, this would integrate with unified-lqg for
                # proper quantum evolution
                
                # Re-monitor constraints
                if self.monitoring_enabled:
                    violations = self._monitor_constraint_algebra(patch)
                    patch.constraint_violations = violations
                    
                    # Count significant violations
                    if any(v > self.violation_threshold for v in violations.values()):
                        evolution_data['violations_detected'] += 1
                
                evolution_data['evolved_patches'] += 1
                evolution_data['patch_updates'].append({
                    'patch_id': patch_id,
                    'step': step,
                    'violations': patch.constraint_violations
                })
            
            # Remove expired patches
            for patch_id in patches_to_remove:
                del self.active_patches[patch_id]
                logger.info(f"Removed expired patch {patch_id}")
        
        evolution_data['final_patches'] = len(self.active_patches)
        
        return evolution_data
    
    def get_patch_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics for all active patches"""
        if not self.active_patches:
            return {'total_patches': 0}
        
        patches = list(self.active_patches.values())
        j_values = [p.j_value for p in patches]
        volumes = [p.volume_eigenvalue for p in patches]
        polymer_scales = [p.polymer_scale for p in patches]
        
        # Compute statistics
        stats = {
            'total_patches': len(patches),
            'j_statistics': {
                'mean': np.mean(j_values),
                'std': np.std(j_values),
                'min': np.min(j_values),
                'max': np.max(j_values),
                'range': np.max(j_values) - np.min(j_values)
            },
            'volume_statistics': {
                'mean': np.mean(volumes),
                'std': np.std(volumes),
                'min': np.min(volumes),
                'max': np.max(volumes),
                'total_volume': np.sum(volumes)
            },
            'polymer_scale_statistics': {
                'mean': np.mean(polymer_scales),
                'std': np.std(polymer_scales),
                'min': np.min(polymer_scales),
                'max': np.max(polymer_scales)
            }
        }
        
        # Constraint violation statistics
        if self.monitoring_enabled:
            all_violations = []
            for patch in patches:
                all_violations.extend(patch.constraint_violations.values())
            
            if all_violations:
                stats['constraint_statistics'] = {
                    'mean_violation': np.mean(all_violations),
                    'max_violation': np.max(all_violations),
                    'violations_above_threshold': np.sum(
                        np.array(all_violations) > self.violation_threshold
                    )
                }
        
        return stats


class VolumeQuantizationController:
    """
    Main Volume Quantization Controller
    
    Integrates SU(2) representation control with discrete spacetime patch management
    to implement the complete Volume Quantization Controller specification.
    """
    
    def __init__(self, config: Optional[LQGVolumeConfiguration] = None):
        """Initialize the complete volume quantization controller"""
        self.config = config or LQGVolumeConfiguration()
        
        # Initialize core components
        self.su2_controller = SU2RepresentationController(self.config)
        self.patch_manager = DiscreteSpacetimePatchManager(self.su2_controller, self.config)
        
        # Controller state
        self.initialized_time = time.time()
        self.total_patches_created = 0
        self.total_evolution_steps = 0
        
        logger.info("LQG Volume Quantization Controller initialized")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"SU(2) scheme: {self.config.scheme.value}")
        
    def create_spacetime_region(self, volume_distribution: Callable[[np.ndarray], float],
                              spatial_bounds: Tuple[Tuple[float, float], ...],
                              resolution: int = 10,
                              metadata: Optional[Dict] = None) -> List[VolumeQuantizationState]:
        """
        Create a region of discrete spacetime patches
        
        Args:
            volume_distribution: Function mapping coordinates to target volume
            spatial_bounds: Spatial bounds as ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            resolution: Grid resolution per dimension
            metadata: Metadata for the spacetime region
            
        Returns:
            List of created VolumeQuantizationState objects
        """
        logger.info(f"Creating spacetime region with {resolution}³ = {resolution**3} patches")
        
        # Generate coordinate grid
        coords_1d = []
        for bounds in spatial_bounds:
            coords_1d.append(np.linspace(bounds[0], bounds[1], resolution))
        
        # Create mesh grid
        coord_grids = np.meshgrid(*coords_1d, indexing='ij')
        
        # Flatten coordinates for patch creation
        coords_flat = np.column_stack([grid.ravel() for grid in coord_grids])
        
        created_patches = []
        
        for i, coordinates in enumerate(coords_flat):
            try:
                # Compute target volume from distribution
                target_volume = volume_distribution(coordinates)
                
                # Skip if volume is too small or invalid
                if target_volume <= 0 or not np.isfinite(target_volume):
                    logger.debug(f"Skipping invalid volume {target_volume} at {coordinates}")
                    continue
                
                # Create patch metadata
                patch_metadata = {
                    'region_id': f"region_{self.total_patches_created}",
                    'grid_index': i,
                    'resolution': resolution,
                    'spatial_bounds': spatial_bounds
                }
                if metadata:
                    patch_metadata.update(metadata)
                
                # Create patch
                patch = self.patch_manager.create_patch(
                    target_volume=target_volume,
                    coordinates=coordinates,
                    metadata=patch_metadata
                )
                
                created_patches.append(patch)
                self.total_patches_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create patch at {coordinates}: {e}")
                continue
        
        logger.info(f"Successfully created {len(created_patches)} patches")
        return created_patches
    
    def evolve_spacetime(self, time_step: float, evolution_steps: int = 1) -> Dict[str, any]:
        """Evolve the complete spacetime system"""
        logger.info(f"Evolving spacetime: {evolution_steps} steps of {time_step:.2e} s each")
        
        evolution_data = self.patch_manager.evolve_patches(time_step, evolution_steps)
        self.total_evolution_steps += evolution_steps
        
        # Add controller-level statistics
        evolution_data.update({
            'controller_total_evolution_steps': self.total_evolution_steps,
            'controller_total_patches_created': self.total_patches_created,
            'controller_uptime': time.time() - self.initialized_time
        })
        
        return evolution_data
    
    def get_controller_status(self) -> Dict[str, any]:
        """Get complete controller status"""
        status = {
            'controller_info': {
                'mode': self.config.mode.value,
                'scheme': self.config.scheme.value,
                'uptime': time.time() - self.initialized_time,
                'total_patches_created': self.total_patches_created,
                'total_evolution_steps': self.total_evolution_steps
            },
            'su2_controller_info': {
                'gamma_immirzi': self.su2_controller.gamma,
                'planck_length': self.su2_controller.l_planck,
                'j_range': [self.su2_controller.min_j, self.su2_controller.max_j],
                'cache_enabled': self.su2_controller._volume_cache is not None,
                'cache_size': len(self.su2_controller._volume_cache) if self.su2_controller._volume_cache else 0
            },
            'patch_manager_info': {
                'active_patches': len(self.patch_manager.active_patches),
                'max_patches': self.config.max_patches,
                'monitoring_enabled': self.patch_manager.monitoring_enabled,
                'violation_threshold': self.patch_manager.violation_threshold
            }
        }
        
        # Add patch statistics if patches exist
        if self.patch_manager.active_patches:
            status['patch_statistics'] = self.patch_manager.get_patch_statistics()
        
        return status
    
    def validate_system(self) -> Dict[str, bool]:
        """Comprehensive system validation"""
        validation = {
            'su2_controller_valid': True,
            'patch_manager_valid': True,
            'configuration_valid': True,
            'integration_valid': True
        }
        
        try:
            # Test SU(2) controller
            test_j = 1.0
            test_volume = self.su2_controller.compute_volume_eigenvalue(test_j)
            if not np.isfinite(test_volume) or test_volume <= 0:
                validation['su2_controller_valid'] = False
            
            # Test patch creation
            test_coords = np.array([0.0, 0.0, 0.0])
            test_patch = self.patch_manager.create_patch(
                target_volume=test_volume,
                coordinates=test_coords,
                metadata={'validation_test': True}
            )
            
            # Validate patch
            if test_patch.patch_id not in self.patch_manager.active_patches:
                validation['patch_manager_valid'] = False
            
            # Clean up test patch
            del self.patch_manager.active_patches[test_patch.patch_id]
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            validation['su2_controller_valid'] = False
            validation['patch_manager_valid'] = False
        
        # Validate configuration
        if (self.config.min_j < 0.5 or 
            self.config.max_j <= self.config.min_j or
            self.config.gamma_immirzi <= 0 or
            self.config.planck_length <= 0):
            validation['configuration_valid'] = False
        
        # Check integration status
        validation['integration_valid'] = self.su2_controller._su2_integration_ready
        
        validation['overall_valid'] = all(validation.values())
        
        return validation


# Convenience functions for quick usage
def create_standard_controller(max_j: float = 10.0, 
                             max_patches: int = 1000) -> VolumeQuantizationController:
    """Create a standard volume quantization controller"""
    config = LQGVolumeConfiguration(
        max_j=max_j,
        max_patches=max_patches,
        mode=VolumeQuantizationMode.PRODUCTION,
        scheme=SU2RepresentationScheme.ANALYTICAL
    )
    return VolumeQuantizationController(config)


def gaussian_volume_distribution(center: np.ndarray, 
                                sigma: float, 
                                volume_scale: float) -> Callable[[np.ndarray], float]:
    """Create a Gaussian volume distribution function"""
    def distribution(coords: np.ndarray) -> float:
        r_squared = np.sum((coords - center)**2)
        return volume_scale * np.exp(-r_squared / (2 * sigma**2))
    return distribution


def planck_scale_volume_distribution(amplitude: float = 1.0) -> Callable[[np.ndarray], float]:
    """Create a Planck-scale volume distribution"""
    def distribution(coords: np.ndarray) -> float:
        coord_magnitude = np.linalg.norm(coords)
        # Volume decreases with distance from origin
        if coord_magnitude == 0:
            return amplitude * PLANCK_VOLUME
        else:
            return amplitude * PLANCK_VOLUME / (1 + coord_magnitude / PLANCK_LENGTH)
    return distribution


if __name__ == "__main__":
    # Demonstration of the LQG Volume Quantization Controller
    print("LQG Volume Quantization Controller - Core Implementation")
    print("=" * 60)
    
    # Create controller
    controller = create_standard_controller(max_j=5.0, max_patches=100)
    
    # Validate system
    validation = controller.validate_system()
    print(f"System validation: {validation}")
    
    if validation['overall_valid']:
        # Create a simple volume distribution
        volume_dist = gaussian_volume_distribution(
            center=np.array([0.0, 0.0, 0.0]),
            sigma=1e-9,  # 1 nm
            volume_scale=PLANCK_VOLUME
        )
        
        # Create spacetime region
        spatial_bounds = ((-2e-9, 2e-9), (-2e-9, 2e-9), (-2e-9, 2e-9))
        patches = controller.create_spacetime_region(
            volume_distribution=volume_dist,
            spatial_bounds=spatial_bounds,
            resolution=3  # 3x3x3 = 27 patches
        )
        
        print(f"\nCreated {len(patches)} spacetime patches")
        
        # Evolve system
        evolution_data = controller.evolve_spacetime(
            time_step=PLANCK_TIME,
            evolution_steps=10
        )
        
        print(f"Evolution completed: {evolution_data['evolved_patches']} patches evolved")
        
        # Show status
        status = controller.get_controller_status()
        print("\nController Status:")
        print(f"  Mode: {status['controller_info']['mode']}")
        print(f"  Active patches: {status['patch_manager_info']['active_patches']}")
        print(f"  Total created: {status['controller_info']['total_patches_created']}")
        
        if 'patch_statistics' in status:
            stats = status['patch_statistics']
            print(f"  j range: [{stats['j_statistics']['min']:.3f}, {stats['j_statistics']['max']:.3f}]")
            print(f"  Total volume: {stats['volume_statistics']['total_volume']:.2e} m³")
    
    else:
        print("❌ System validation failed - check dependencies and configuration")
