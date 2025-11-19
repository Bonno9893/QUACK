"""
Lambda Parameter Optimization Module

This module provides functionality for optimizing the lambda penalty parameter
in QUBO formulations. The lambda parameter controls the strength of constraint
enforcement and is crucial for obtaining feasible solutions.

The optimization process uses an adaptive approach that:
1. Tests multiple lambda values
2. Evaluates solution feasibility and quality
3. Selects the optimal trade-off

Author: QUACK Project Team
Date: 2024
License: MIT
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import json

from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

logger = logging.getLogger(__name__)


@dataclass
class LambdaSearchConfig:
    """Configuration for lambda parameter search."""
    lambda_range: Tuple[float, float] = (0.1, 10.0)
    num_samples: int = 20
    search_method: str = 'adaptive'  # 'adaptive', 'grid', 'binary'
    convergence_threshold: float = 0.95  # Feasibility rate threshold
    max_iterations: int = 50
    early_stopping: bool = True
    num_reads_per_test: int = 100  # SA reads for each lambda test


class LambdaOptimizer:
    """
    Optimizer for the lambda penalty parameter in QUBO formulations.
    
    This class implements various strategies for finding the optimal lambda
    value that balances solution quality with constraint satisfaction.
    """
    
    def __init__(
        self,
        solver: str = 'simulated_annealing',
        config: Optional[LambdaSearchConfig] = None
    ):
        """
        Initialize the lambda optimizer.
        
        Args:
            solver (str): Solver to use for testing ('simulated_annealing' or 'exact')
            config (LambdaSearchConfig, optional): Search configuration
        """
        self.solver_type = solver
        self.config = config or LambdaSearchConfig()
        
        # Initialize test solver
        if solver == 'simulated_annealing':
            self.test_sampler = SimulatedAnnealingSampler()
        else:
            raise ValueError(f"Unsupported solver: {solver}")
        
        # Cache for tested lambda values
        self.lambda_cache = {}
        
        logger.info(f"Lambda optimizer initialized with {solver} solver")
    
    def optimize_for_instance(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        candidate_points: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal lambda for a specific problem instance.
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            seed_cluster (np.ndarray): Seed cluster indices
            n_expand (int): Number of points to expand
            candidate_points (np.ndarray, optional): Candidate points
        
        Returns:
            Tuple[float, Dict]: (optimal_lambda, optimization_metrics)
        """
        logger.info(f"Optimizing lambda for instance: seed_size={len(seed_cluster)}, "
                   f"expand={n_expand}")
        
        # Select optimization method
        if self.config.search_method == 'adaptive':
            return self._adaptive_search(
                distance_matrix, seed_cluster, n_expand, candidate_points
            )
        elif self.config.search_method == 'grid':
            return self._grid_search(
                distance_matrix, seed_cluster, n_expand, candidate_points
            )
        elif self.config.search_method == 'binary':
            return self._binary_search(
                distance_matrix, seed_cluster, n_expand, candidate_points
            )
        else:
            raise ValueError(f"Unknown search method: {self.config.search_method}")
    
    def _adaptive_search(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        candidate_points: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Adaptive lambda search that adjusts based on problem characteristics.
        
        This method starts with an estimate based on the distance scale and
        refines the lambda value based on feasibility feedback.
        
        Args:
            distance_matrix: Distance matrix
            seed_cluster: Seed cluster
            n_expand: Expansion size
            candidate_points: Candidates
        
        Returns:
            Tuple[float, Dict]: Optimal lambda and metrics
        """
        # Estimate initial lambda based on distance scale
        distance_scale = self._estimate_distance_scale(distance_matrix)
        lambda_current = distance_scale
        
        logger.info(f"Starting adaptive search with initial λ = {lambda_current:.3f}")
        
        # Track tested values and results
        tested_lambdas = []
        results = []
        
        # Phase 1: Find feasible region
        feasible_found = False
        iteration = 0
        
        while iteration < self.config.max_iterations:
            # Test current lambda
            metrics = self._test_lambda(
                lambda_current,
                distance_matrix,
                seed_cluster,
                n_expand,
                candidate_points
            )
            
            tested_lambdas.append(lambda_current)
            results.append(metrics)
            
            logger.debug(f"Iteration {iteration}: λ={lambda_current:.3f}, "
                        f"feasibility={metrics['feasibility_rate']:.2%}")
            
            # Check if we found a good lambda
            if metrics['feasibility_rate'] >= self.config.convergence_threshold:
                feasible_found = True
                if self.config.early_stopping:
                    break
            
            # Adjust lambda based on feasibility
            if metrics['avg_selected'] < n_expand:
                # Too few points selected, decrease penalty
                lambda_current *= 0.8
            elif metrics['avg_selected'] > n_expand:
                # Too many points selected, increase penalty
                lambda_current *= 1.2
            else:
                # Right number but not consistent, fine-tune
                if metrics['feasibility_rate'] < 0.5:
                    lambda_current *= 1.1
                else:
                    lambda_current *= 0.95
            
            # Keep within bounds
            lambda_current = np.clip(
                lambda_current,
                self.config.lambda_range[0],
                self.config.lambda_range[1]
            )
            
            iteration += 1
        
        # Phase 2: Refine among feasible lambdas
        if feasible_found:
            feasible_indices = [
                i for i, r in enumerate(results)
                if r['feasibility_rate'] >= 0.8
            ]
            
            if feasible_indices:
                # Among feasible, choose with best solution quality
                best_idx = min(
                    feasible_indices,
                    key=lambda i: results[i]['avg_energy']
                )
                optimal_lambda = tested_lambdas[best_idx]
                best_metrics = results[best_idx]
            else:
                # Use the one with highest feasibility
                best_idx = max(
                    range(len(results)),
                    key=lambda i: results[i]['feasibility_rate']
                )
                optimal_lambda = tested_lambdas[best_idx]
                best_metrics = results[best_idx]
        else:
            # No feasible lambda found, return best attempt
            logger.warning("No feasible lambda found in adaptive search")
            best_idx = max(
                range(len(results)),
                key=lambda i: results[i]['feasibility_rate']
            )
            optimal_lambda = tested_lambdas[best_idx]
            best_metrics = results[best_idx]
        
        # Compile optimization report
        optimization_metrics = {
            'optimal_lambda': optimal_lambda,
            'feasibility_rate': best_metrics['feasibility_rate'],
            'avg_energy': best_metrics['avg_energy'],
            'iterations': iteration,
            'tested_lambdas': tested_lambdas,
            'all_results': results
        }
        
        logger.info(f"Adaptive search complete: optimal λ = {optimal_lambda:.3f}, "
                   f"feasibility = {best_metrics['feasibility_rate']:.2%}")
        
        return optimal_lambda, optimization_metrics
    
    def _grid_search(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        candidate_points: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Grid search over lambda values.
        
        Tests evenly spaced lambda values across the specified range.
        
        Args:
            distance_matrix: Distance matrix
            seed_cluster: Seed cluster
            n_expand: Expansion size
            candidate_points: Candidates
        
        Returns:
            Tuple[float, Dict]: Optimal lambda and metrics
        """
        lambda_values = np.linspace(
            self.config.lambda_range[0],
            self.config.lambda_range[1],
            self.config.num_samples
        )
        
        logger.info(f"Grid search: testing {len(lambda_values)} lambda values")
        
        results = []
        for i, lambda_val in enumerate(lambda_values):
            logger.debug(f"Testing λ = {lambda_val:.3f} ({i+1}/{len(lambda_values)})")
            
            metrics = self._test_lambda(
                lambda_val,
                distance_matrix,
                seed_cluster,
                n_expand,
                candidate_points
            )
            
            results.append({
                'lambda': lambda_val,
                **metrics
            })
            
            # Early stopping if perfect feasibility found
            if self.config.early_stopping and metrics['feasibility_rate'] >= 0.99:
                logger.info(f"Early stopping: perfect feasibility at λ = {lambda_val:.3f}")
                break
        
        # Select best lambda
        # First filter by feasibility, then by energy
        feasible_results = [
            r for r in results 
            if r['feasibility_rate'] >= 0.5
        ]
        
        if feasible_results:
            best_result = min(feasible_results, key=lambda r: r['avg_energy'])
        else:
            # No feasible solution, pick best feasibility
            best_result = max(results, key=lambda r: r['feasibility_rate'])
        
        optimal_lambda = best_result['lambda']
        
        optimization_metrics = {
            'optimal_lambda': optimal_lambda,
            'feasibility_rate': best_result['feasibility_rate'],
            'avg_energy': best_result['avg_energy'],
            'grid_size': len(lambda_values),
            'all_results': results
        }
        
        logger.info(f"Grid search complete: optimal λ = {optimal_lambda:.3f}")
        
        return optimal_lambda, optimization_metrics
    
    def _binary_search(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        candidate_points: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Binary search for optimal lambda value.
        
        This method performs binary search to find the lambda value where
        the solution transitions from infeasible to feasible.
        
        Args:
            distance_matrix: Distance matrix
            seed_cluster: Seed cluster
            n_expand: Expansion size
            candidate_points: Candidates
        
        Returns:
            Tuple[float, Dict]: Optimal lambda and metrics
        """
        lambda_low = self.config.lambda_range[0]
        lambda_high = self.config.lambda_range[1]
        
        logger.info(f"Binary search: λ ∈ [{lambda_low:.3f}, {lambda_high:.3f}]")
        
        tested_lambdas = []
        results = []
        iteration = 0
        
        while iteration < self.config.max_iterations and (lambda_high - lambda_low) > 0.01:
            lambda_mid = (lambda_low + lambda_high) / 2
            
            metrics = self._test_lambda(
                lambda_mid,
                distance_matrix,
                seed_cluster,
                n_expand,
                candidate_points
            )
            
            tested_lambdas.append(lambda_mid)
            results.append({'lambda': lambda_mid, **metrics})
            
            logger.debug(f"Binary search iteration {iteration}: λ={lambda_mid:.3f}, "
                        f"avg_selected={metrics['avg_selected']:.1f}/{n_expand}")
            
            # Adjust search range based on selection behavior
            if metrics['avg_selected'] < n_expand:
                # Too few selected, need weaker penalty
                lambda_high = lambda_mid
            elif metrics['avg_selected'] > n_expand:
                # Too many selected, need stronger penalty
                lambda_low = lambda_mid
            else:
                # Right number selected, check consistency
                if metrics['feasibility_rate'] < 0.9:
                    # Need fine-tuning
                    if iteration % 2 == 0:
                        lambda_low = lambda_mid * 0.99
                    else:
                        lambda_high = lambda_mid * 1.01
                else:
                    # Found good lambda
                    break
            
            iteration += 1
        
        # Select best from tested values
        if results:
            feasible_results = [
                r for r in results 
                if r['feasibility_rate'] >= 0.5
            ]
            
            if feasible_results:
                best_result = min(feasible_results, key=lambda r: r['avg_energy'])
                optimal_lambda = best_result['lambda']
            else:
                best_result = max(results, key=lambda r: r['feasibility_rate'])
                optimal_lambda = best_result['lambda']
        else:
            optimal_lambda = (lambda_low + lambda_high) / 2
            best_result = {'feasibility_rate': 0, 'avg_energy': float('inf')}
        
        optimization_metrics = {
            'optimal_lambda': optimal_lambda,
            'feasibility_rate': best_result['feasibility_rate'],
            'avg_energy': best_result['avg_energy'],
            'iterations': iteration,
            'tested_lambdas': tested_lambdas
        }
        
        logger.info(f"Binary search complete: optimal λ = {optimal_lambda:.3f}")
        
        return optimal_lambda, optimization_metrics
    
    def _test_lambda(
        self,
        lambda_value: float,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        candidate_points: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Test a specific lambda value and return metrics.
        
        Args:
            lambda_value: Lambda to test
            distance_matrix: Distance matrix
            seed_cluster: Seed cluster
            n_expand: Expansion size
            candidate_points: Candidates
        
        Returns:
            Dict with test metrics
        """
        # Check cache first
        cache_key = (
            lambda_value,
            hash(distance_matrix.tobytes()),
            hash(seed_cluster.tobytes()),
            n_expand
        )
        
        if cache_key in self.lambda_cache:
            return self.lambda_cache[cache_key]
        
        # Create QUBO with given lambda
        from ..optimization.qubo_formulation import QUBOFormulator
        formulator = QUBOFormulator(distance_matrix)
        
        bqm = formulator.create_cluster_expansion_qubo(
            seed_cluster=seed_cluster,
            target_size=n_expand,
            lambda_penalty=lambda_value,
            candidate_points=candidate_points
        )
        
        # Run test solver
        sampleset = self.test_sampler.sample(
            bqm,
            num_reads=self.config.num_reads_per_test,
            num_sweeps=100,  # Quick test
            seed=42
        )
        
        # Analyze results
        feasible_count = 0
        total_selected = []
        energies = []
        
        for sample in sampleset:
            selected = sum(sample.sample.values())
            total_selected.append(selected)
            energies.append(sample.energy)
            
            if selected == n_expand:
                feasible_count += sample.num_occurrences
        
        total_occurrences = sum(s.num_occurrences for s in sampleset)
        feasibility_rate = feasible_count / total_occurrences if total_occurrences > 0 else 0
        
        metrics = {
            'feasibility_rate': feasibility_rate,
            'avg_selected': np.mean(total_selected),
            'std_selected': np.std(total_selected),
            'avg_energy': np.mean(energies),
            'min_energy': min(energies),
            'unique_solutions': len(set(str(s.sample) for s in sampleset))
        }
        
        # Cache result
        self.lambda_cache[cache_key] = metrics
        
        return metrics
    
    def _estimate_distance_scale(self, distance_matrix: np.ndarray) -> float:
        """
        Estimate appropriate lambda scale based on distance matrix.
        
        Args:
            distance_matrix: Pairwise distances
        
        Returns:
            float: Estimated lambda scale
        """
        # Get non-zero distances
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        distances = distances[distances > 0]
        
        if len(distances) == 0:
            return 1.0
        
        # Use median distance as scale estimate
        # Lambda should be comparable to typical distance values
        median_dist = np.median(distances)
        
        # Adjust based on problem size (larger problems need stronger penalties)
        n_points = distance_matrix.shape[0]
        size_factor = 1 + np.log10(n_points)
        
        estimated_lambda = median_dist * size_factor
        
        # Keep within configured range
        estimated_lambda = np.clip(
            estimated_lambda,
            self.config.lambda_range[0],
            self.config.lambda_range[1]
        )
        
        logger.debug(f"Estimated λ scale: {estimated_lambda:.3f} "
                    f"(median dist: {median_dist:.3f}, size factor: {size_factor:.2f})")
        
        return estimated_lambda
    
    def save_optimization_history(self, filepath: str):
        """
        Save optimization history to file.
        
        Args:
            filepath: Path to save JSON file
        """
        history = {
            'config': {
                'lambda_range': self.config.lambda_range,
                'num_samples': self.config.num_samples,
                'search_method': self.config.search_method
            },
            'cache_size': len(self.lambda_cache),
            'cached_results': [
                {
                    'lambda': k[0],
                    'n_expand': k[3],
                    'metrics': v
                }
                for k, v in list(self.lambda_cache.items())[:100]  # Limit to 100 entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Saved optimization history to {filepath}")
