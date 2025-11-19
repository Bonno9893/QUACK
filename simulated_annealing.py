"""
Simulated Annealing Solver Module

This module provides a classical simulated annealing solver for clustering
problems, serving as both a standalone solver and a tool for parameter
optimization in the quantum pipeline.

The implementation uses D-Wave's neal package for consistency with the
quantum formulation, allowing direct comparison of results.

Author: QUACK Project Team
Date: 2024
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
import dimod

logger = logging.getLogger(__name__)


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing solver."""
    num_reads: int = 1000
    num_sweeps: int = 1000
    beta_range: Tuple[float, float] = (0.1, 10.0)
    beta_schedule_type: str = 'geometric'
    seed: Optional[int] = 42
    initial_states_generator: str = 'random'


class SimulatedAnnealingSolver:
    """
    Simulated Annealing solver for clustering problems.
    
    This solver provides a classical baseline for comparison with quantum
    annealing results. It uses the same QUBO formulation as the quantum
    solver for direct comparison.
    """
    
    def __init__(self, config: Optional[SAConfig] = None):
        """
        Initialize the Simulated Annealing solver.
        
        Args:
            config (SAConfig, optional): Configuration object. If None, uses defaults.
        """
        self.config = config or SAConfig()
        self.sampler = SimulatedAnnealingSampler()
        
        logger.info(f"Initialized SA solver with {self.config.num_reads} reads")
    
    def solve(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        lambda_penalty: float,
        candidate_points: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Solve cluster expansion problem using Simulated Annealing.
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            seed_cluster (np.ndarray): Indices of seed cluster points
            n_expand (int): Number of points to add
            lambda_penalty (float): Penalty parameter
            candidate_points (np.ndarray, optional): Candidate points for expansion
        
        Returns:
            Dict containing solution details:
                - 'best_sample': Binary solution vector
                - 'energy': Solution energy
                - 'selected_indices': Selected point indices
                - 'feasible': Whether solution is feasible
                - 'num_occurrences': Times best solution was found
                - 'convergence_data': Energy evolution over iterations
        """
        # Create QUBO formulation
        from ..optimization.qubo_formulation import QUBOFormulator
        formulator = QUBOFormulator(distance_matrix)
        
        bqm = formulator.create_cluster_expansion_qubo(
            seed_cluster=seed_cluster,
            target_size=n_expand,
            lambda_penalty=lambda_penalty,
            candidate_points=candidate_points
        )
        
        # Determine candidates
        if candidate_points is None:
            all_points = set(range(distance_matrix.shape[0]))
            seed_set = set(seed_cluster)
            candidate_points = np.array(list(all_points - seed_set))
        
        # Run Simulated Annealing
        logger.info(f"Running SA with {self.config.num_reads} reads, "
                   f"λ={lambda_penalty:.3f}")
        
        sampleset = self.sampler.sample(
            bqm,
            num_reads=self.config.num_reads,
            num_sweeps=self.config.num_sweeps,
            beta_range=self.config.beta_range,
            beta_schedule_type=self.config.beta_schedule_type,
            seed=self.config.seed,
            initial_states_generator=self.config.initial_states_generator
        )
        
        # Process results
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Extract selected indices
        selected_vars = [var for var, val in best_sample.items() if val == 1]
        selected_indices = np.array([
            candidate_points[int(var.split('_')[1])] 
            for var in selected_vars
        ])
        
        # Check feasibility
        is_feasible = (len(selected_indices) == n_expand)
        
        # Calculate solution statistics
        solution_stats = self._calculate_solution_statistics(
            sampleset, n_expand
        )
        
        logger.info(f"SA solution: energy={best_energy:.3f}, "
                   f"selected={len(selected_indices)}/{n_expand}, "
                   f"feasible={is_feasible}")
        
        return {
            'best_sample': best_sample,
            'energy': best_energy,
            'selected_indices': selected_indices,
            'feasible': is_feasible,
            'num_occurrences': sampleset.first.num_occurrences,
            'solution_stats': solution_stats,
            'all_energies': [s.energy for s in sampleset],
            'convergence_data': self._extract_convergence_data(sampleset)
        }
    
    def solve_global_clustering(
        self,
        distance_matrix: np.ndarray,
        n_clusters: int,
        lambda_penalty: float
    ) -> Dict[str, Any]:
        """
        Solve global clustering problem (Algorithm 2).
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            n_clusters (int): Number of clusters
            lambda_penalty (float): Penalty for constraint violations
        
        Returns:
            Dict containing clustering solution
        """
        from ..optimization.qubo_formulation import QUBOFormulator
        formulator = QUBOFormulator(distance_matrix)
        
        bqm = formulator.create_global_clustering_qubo(
            n_clusters=n_clusters,
            lambda_penalty=lambda_penalty
        )
        
        logger.info(f"Solving global clustering: {distance_matrix.shape[0]} points, "
                   f"{n_clusters} clusters")
        
        # Run SA
        sampleset = self.sampler.sample(
            bqm,
            num_reads=self.config.num_reads,
            num_sweeps=self.config.num_sweeps * 2,  # More sweeps for larger problem
            beta_range=self.config.beta_range,
            beta_schedule_type=self.config.beta_schedule_type,
            seed=self.config.seed
        )
        
        # Extract clustering assignment
        best_sample = sampleset.first.sample
        n_points = distance_matrix.shape[0]
        
        clustering = np.full(n_points, -1)
        for i in range(n_points):
            for k in range(n_clusters):
                if best_sample.get(f"x_{i}_{k}", 0) == 1:
                    clustering[i] = k
                    break
        
        # Check feasibility (all points assigned to exactly one cluster)
        is_feasible = np.all(clustering >= 0)
        
        # Calculate clustering cost
        total_cost = self._calculate_clustering_cost(
            distance_matrix, clustering
        )
        
        logger.info(f"Global clustering: cost={total_cost:.3f}, "
                   f"feasible={is_feasible}")
        
        return {
            'clustering': clustering,
            'energy': sampleset.first.energy,
            'cost': total_cost,
            'feasible': is_feasible,
            'cluster_sizes': np.bincount(clustering[clustering >= 0])
        }
    
    def _calculate_solution_statistics(
        self,
        sampleset,
        target_size: int
    ) -> Dict[str, Any]:
        """
        Calculate statistics about solution distribution.
        
        Args:
            sampleset: SA sample set
            target_size: Expected number of selected points
        
        Returns:
            Dict with solution statistics
        """
        feasible_count = 0
        energy_distribution = []
        size_distribution = []
        
        for sample in sampleset:
            # Count selected variables
            selected = sum(sample.sample.values())
            size_distribution.append(selected)
            
            # Check feasibility
            if selected == target_size:
                feasible_count += 1
                energy_distribution.append(sample.energy)
        
        feasibility_rate = feasible_count / len(sampleset)
        
        stats = {
            'feasibility_rate': feasibility_rate,
            'avg_energy_feasible': np.mean(energy_distribution) if energy_distribution else None,
            'std_energy_feasible': np.std(energy_distribution) if energy_distribution else None,
            'size_distribution': np.bincount(size_distribution),
            'unique_solutions': len(set(str(s.sample) for s in sampleset))
        }
        
        return stats
    
    def _calculate_clustering_cost(
        self,
        distance_matrix: np.ndarray,
        clustering: np.ndarray
    ) -> float:
        """
        Calculate total intra-cluster distance.
        
        Args:
            distance_matrix: Pairwise distances
            clustering: Cluster assignments
        
        Returns:
            float: Total cost
        """
        cost = 0.0
        unique_clusters = np.unique(clustering[clustering >= 0])
        
        for cluster_id in unique_clusters:
            cluster_points = np.where(clustering == cluster_id)[0]
            
            # Sum pairwise distances within cluster
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    cost += distance_matrix[cluster_points[i], cluster_points[j]]
        
        return cost
    
    def _extract_convergence_data(self, sampleset) -> Dict[str, List]:
        """
        Extract convergence information from sample set.
        
        Args:
            sampleset: SA sample set
        
        Returns:
            Dict with convergence metrics
        """
        # Group samples by energy level
        energy_counts = {}
        for sample in sampleset:
            energy = round(sample.energy, 3)
            if energy not in energy_counts:
                energy_counts[energy] = 0
            energy_counts[energy] += sample.num_occurrences
        
        # Sort by energy
        sorted_energies = sorted(energy_counts.items())
        
        return {
            'energy_levels': [e for e, _ in sorted_energies],
            'counts': [c for _, c in sorted_energies],
            'best_energy': sampleset.first.energy,
            'worst_energy': max(s.energy for s in sampleset),
            'energy_gap': max(s.energy for s in sampleset) - sampleset.first.energy
        }
    
    def parameter_sweep(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        lambda_range: Tuple[float, float],
        n_lambdas: int = 10
    ) -> Dict[float, Dict]:
        """
        Perform parameter sweep to find optimal lambda.
        
        This method tests multiple lambda values to find the one that
        produces the best trade-off between solution quality and feasibility.
        
        Args:
            distance_matrix: Distance matrix
            seed_cluster: Seed cluster indices
            n_expand: Expansion size
            lambda_range: Range of lambda values to test
            n_lambdas: Number of lambda values to test
        
        Returns:
            Dict mapping lambda values to solution metrics
        """
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], n_lambdas)
        results = {}
        
        logger.info(f"Starting parameter sweep: λ ∈ [{lambda_range[0]:.2f}, "
                   f"{lambda_range[1]:.2f}] with {n_lambdas} samples")
        
        for lambda_val in lambda_values:
            logger.info(f"Testing λ = {lambda_val:.3f}")
            
            solution = self.solve(
                distance_matrix=distance_matrix,
                seed_cluster=seed_cluster,
                n_expand=n_expand,
                lambda_penalty=lambda_val
            )
            
            results[lambda_val] = {
                'energy': solution['energy'],
                'feasible': solution['feasible'],
                'selected_count': len(solution['selected_indices']),
                'feasibility_rate': solution['solution_stats']['feasibility_rate']
            }
        
        # Find best lambda (highest feasibility rate with lowest energy)
        feasible_lambdas = [
            lam for lam, res in results.items() 
            if res['feasibility_rate'] > 0.5
        ]
        
        if feasible_lambdas:
            # Among feasible ones, choose with lowest energy
            best_lambda = min(
                feasible_lambdas,
                key=lambda l: results[l]['energy']
            )
            
            logger.info(f"Best λ found: {best_lambda:.3f} with "
                       f"feasibility rate {results[best_lambda]['feasibility_rate']:.2%}")
        else:
            # No good lambda found, pick middle value
            best_lambda = (lambda_range[0] + lambda_range[1]) / 2
            logger.warning(f"No feasible λ found, using default: {best_lambda:.3f}")
        
        results['best_lambda'] = best_lambda
        
        return results
