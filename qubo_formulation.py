"""
QUBO Formulation Module for Clustering Problems

This module provides the core functionality for formulating clustering problems
as Quadratic Unconstrained Binary Optimization (QUBO) models, specifically
designed for quantum annealing on D-Wave systems.

The main focus is on Algorithm 1 (Cluster Expansion), which selects exactly T
points to add to an existing cluster seed while minimizing intra-cluster distances.

Mathematical Formulation:
    min Σ(i,j) d_ij * x_i * x_j + λ₂ * (Σx_i - T)²
    
Where:
    - d_ij: Distance between points i and j
    - x_i: Binary decision variable (1 if point i is selected, 0 otherwise)
    - T: Target number of points to select (cardinality constraint)
    - λ₂: Penalty parameter for constraint violation

Author: QUACK Project Team
Date: 2024
License: MIT
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from dimod import BinaryQuadraticModel, QuadraticModel
import logging

# Configure logging
logger = logging.getLogger(__name__)


class QUBOFormulator:
    """
    A class for formulating clustering problems as QUBO models.
    
    This formulator creates Binary Quadratic Models (BQM) suitable for
    quantum annealing, with special attention to cardinality constraints
    that are common in clustering applications.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize the QUBO formulator with a distance matrix.
        
        Args:
            distance_matrix (np.ndarray): Square symmetric matrix of pairwise distances
                                         between all points in the dataset.
        
        Raises:
            ValueError: If the distance matrix is not square or symmetric.
        """
        self._validate_distance_matrix(distance_matrix)
        self.distance_matrix = distance_matrix
        self.n_points = distance_matrix.shape[0]
        
        logger.info(f"Initialized QUBO formulator for {self.n_points} points")
    
    def _validate_distance_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate that the distance matrix is properly formatted.
        
        Args:
            matrix (np.ndarray): Matrix to validate
            
        Raises:
            ValueError: If matrix is not square or not symmetric
        """
        if matrix.ndim != 2:
            raise ValueError("Distance matrix must be 2-dimensional")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        
        # Check symmetry (with small tolerance for numerical errors)
        if not np.allclose(matrix, matrix.T, rtol=1e-10):
            raise ValueError("Distance matrix must be symmetric")
        
        # Check non-negative distances
        if np.any(matrix < 0):
            logger.warning("Distance matrix contains negative values")
    
    def create_cluster_expansion_qubo(
        self,
        seed_cluster: np.ndarray,
        target_size: int,
        lambda_penalty: float,
        candidate_points: Optional[np.ndarray] = None
    ) -> BinaryQuadraticModel:
        """
        Create a QUBO model for the cluster expansion problem.
        
        This formulation aims to select exactly 'target_size' points from the
        candidate set that are closest to the seed cluster, minimizing the
        total intra-cluster distance.
        
        Args:
            seed_cluster (np.ndarray): Indices of points already in the cluster
            target_size (int): Number of points to add to the cluster
            lambda_penalty (float): Penalty weight for cardinality constraint
            candidate_points (np.ndarray, optional): Indices of candidate points.
                                                     If None, all non-seed points
                                                     are candidates.
        
        Returns:
            BinaryQuadraticModel: The QUBO model ready for solving
        
        Example:
            >>> formulator = QUBOFormulator(distance_matrix)
            >>> seed = np.array([0, 1, 2])  # Starting cluster
            >>> bqm = formulator.create_cluster_expansion_qubo(
            ...     seed_cluster=seed,
            ...     target_size=5,  # Add 5 more points
            ...     lambda_penalty=2.0
            ... )
        """
        # Determine candidate points (all points not in seed cluster)
        if candidate_points is None:
            all_points = set(range(self.n_points))
            seed_set = set(seed_cluster)
            candidate_points = np.array(list(all_points - seed_set))
        
        n_candidates = len(candidate_points)
        
        logger.info(f"Creating QUBO for selecting {target_size} from "
                   f"{n_candidates} candidates to expand cluster of size {len(seed_cluster)}")
        
        # Initialize the Binary Quadratic Model
        bqm = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
        
        # === 1. Distance minimization terms ===
        # Add quadratic terms for distances between selected points
        for i in range(n_candidates):
            for j in range(i + 1, n_candidates):
                point_i = candidate_points[i]
                point_j = candidate_points[j]
                distance = self.distance_matrix[point_i, point_j]
                
                # Add interaction term: selecting both i and j incurs their distance
                bqm.add_quadratic(f"x_{i}", f"x_{j}", distance)
        
        # Add linear terms for distances to seed cluster
        for i in range(n_candidates):
            point_i = candidate_points[i]
            distance_to_seed = sum(
                self.distance_matrix[point_i, seed_point]
                for seed_point in seed_cluster
            )
            bqm.add_linear(f"x_{i}", distance_to_seed)
        
        # === 2. Cardinality constraint enforcement ===
        # Penalty term: λ₂ * (Σx_i - T)²
        # Expanded: λ₂ * (ΣΣ x_i*x_j - 2T*Σx_i + T²)
        
        # Quadratic penalty terms
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    bqm.add_quadratic(f"x_{i}", f"x_{j}", lambda_penalty)
                else:
                    # Self-interaction (x_i * x_i = x_i for binary variables)
                    bqm.add_linear(f"x_{i}", lambda_penalty)
        
        # Linear penalty terms
        for i in range(n_candidates):
            bqm.add_linear(f"x_{i}", -2 * lambda_penalty * target_size)
        
        # Constant term (doesn't affect optimization but needed for energy calculation)
        bqm.offset += lambda_penalty * target_size * target_size
        
        logger.info(f"QUBO model created with {len(bqm.variables)} variables and "
                   f"{len(bqm.quadratic)} quadratic terms")
        
        return bqm
    
    def create_global_clustering_qubo(
        self,
        n_clusters: int,
        lambda_penalty: float
    ) -> BinaryQuadraticModel:
        """
        Create a QUBO model for global clustering (Algorithm 2).
        
        This formulation assigns all points to one of k clusters, ensuring
        each point belongs to exactly one cluster while minimizing total
        intra-cluster distances.
        
        Args:
            n_clusters (int): Number of clusters to form
            lambda_penalty (float): Penalty weight for assignment constraints
        
        Returns:
            BinaryQuadraticModel: The QUBO model for global clustering
        
        Note:
            This uses binary variables x_{i,k} where x_{i,k} = 1 means
            point i is assigned to cluster k.
        """
        bqm = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
        
        logger.info(f"Creating global clustering QUBO for {self.n_points} points "
                   f"into {n_clusters} clusters")
        
        # === 1. Distance minimization within clusters ===
        for k in range(n_clusters):
            for i in range(self.n_points):
                for j in range(i + 1, self.n_points):
                    # Cost of having both points i and j in cluster k
                    var_i = f"x_{i}_{k}"
                    var_j = f"x_{j}_{k}"
                    bqm.add_quadratic(var_i, var_j, self.distance_matrix[i, j])
        
        # === 2. Unique assignment constraint ===
        # Each point must be assigned to exactly one cluster
        # Penalty: λ * (Σ_k x_{i,k} - 1)² for each point i
        
        for i in range(self.n_points):
            vars_i = [f"x_{i}_{k}" for k in range(n_clusters)]
            
            # Add quadratic penalty terms for constraint violation
            for k1 in range(n_clusters):
                for k2 in range(k1 + 1, n_clusters):
                    # Penalize assigning point i to multiple clusters
                    bqm.add_quadratic(vars_i[k1], vars_i[k2], 2 * lambda_penalty)
            
            # Add linear terms
            for var in vars_i:
                bqm.add_linear(var, -2 * lambda_penalty)
            
            # Add constant term
            bqm.offset += lambda_penalty
        
        logger.info(f"Global clustering QUBO created with {len(bqm.variables)} variables")
        
        return bqm
    
    def get_energy_breakdown(
        self,
        solution: Dict[str, int],
        bqm: BinaryQuadraticModel
    ) -> Dict[str, float]:
        """
        Calculate the energy breakdown for a given solution.
        
        This helps understand how much each component (distance vs. penalty)
        contributes to the total energy.
        
        Args:
            solution (Dict[str, int]): Binary solution dictionary
            bqm (BinaryQuadraticModel): The QUBO model
        
        Returns:
            Dict[str, float]: Energy breakdown with keys:
                - 'total': Total energy
                - 'distance': Energy from distance terms
                - 'penalty': Energy from constraint penalties
        """
        total_energy = bqm.energy(solution)
        
        # Calculate distance contribution
        # (This is approximate - assumes we can identify penalty terms by magnitude)
        distance_energy = 0.0
        penalty_energy = 0.0
        
        # Linear terms
        for var, coeff in bqm.linear.items():
            if var in solution:
                contribution = coeff * solution[var]
                # Heuristic: large negative coefficients are likely penalties
                if coeff < -1.0:
                    penalty_energy += contribution
                else:
                    distance_energy += contribution
        
        # Quadratic terms
        for (var1, var2), coeff in bqm.quadratic.items():
            if var1 in solution and var2 in solution:
                contribution = coeff * solution[var1] * solution[var2]
                # Heuristic: large positive coefficients are likely penalties
                if coeff > 1.0:
                    penalty_energy += contribution
                else:
                    distance_energy += contribution
        
        return {
            'total': total_energy,
            'distance': distance_energy,
            'penalty': penalty_energy,
            'offset': bqm.offset
        }
    
    def validate_solution(
        self,
        solution: Dict[str, int],
        target_size: int
    ) -> Tuple[bool, int]:
        """
        Validate if a solution satisfies the cardinality constraint.
        
        Args:
            solution (Dict[str, int]): Binary solution dictionary
            target_size (int): Expected number of selected points
        
        Returns:
            Tuple[bool, int]: (is_valid, actual_size)
        """
        selected_count = sum(solution.values())
        is_valid = (selected_count == target_size)
        
        if not is_valid:
            logger.warning(f"Solution selects {selected_count} points, "
                          f"expected {target_size}")
        
        return is_valid, selected_count


def create_distance_aware_qubo(
    distance_matrix: np.ndarray,
    n_select: int,
    lambda_start: float = 1.0,
    lambda_end: float = 10.0,
    adaptive: bool = True
) -> Tuple[BinaryQuadraticModel, float]:
    """
    Create a QUBO with adaptive lambda parameter selection.
    
    This function automatically determines a good lambda value based on
    the scale of distances in the problem.
    
    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix
        n_select (int): Number of points to select
        lambda_start (float): Starting lambda value for search
        lambda_end (float): Maximum lambda value to consider
        adaptive (bool): Whether to use adaptive lambda selection
    
    Returns:
        Tuple[BinaryQuadraticModel, float]: (QUBO model, selected lambda value)
    """
    formulator = QUBOFormulator(distance_matrix)
    
    if adaptive:
        # Estimate good lambda based on distance scale
        avg_distance = np.mean(distance_matrix[distance_matrix > 0])
        std_distance = np.std(distance_matrix[distance_matrix > 0])
        
        # Lambda should be comparable to the distance scale
        estimated_lambda = avg_distance + std_distance
        
        logger.info(f"Adaptive lambda selection: estimated λ = {estimated_lambda:.3f}")
        
        # Create QUBO with estimated lambda
        seed = np.array([])  # Empty seed for pure selection problem
        bqm = formulator.create_cluster_expansion_qubo(
            seed_cluster=seed,
            target_size=n_select,
            lambda_penalty=estimated_lambda
        )
        
        return bqm, estimated_lambda
    else:
        # Use provided lambda value (middle of range)
        lambda_value = (lambda_start + lambda_end) / 2
        seed = np.array([])
        
        bqm = formulator.create_cluster_expansion_qubo(
            seed_cluster=seed,
            target_size=n_select,
            lambda_penalty=lambda_value
        )
        
        return bqm, lambda_value
