"""
D-Wave Quantum Annealing Solver Module

This module provides the interface for solving clustering problems using
D-Wave's quantum annealing hardware. It handles the complete pipeline from
QUBO formulation to quantum execution and result processing.

Key Features:
    - Automatic embedding to quantum hardware topology
    - Configurable annealing parameters
    - Chain strength optimization
    - Result post-processing and validation
    - Performance metrics collection

Author: QUACK Project Team
Date: 2024
License: MIT
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.embedding import find_embedding
import minorminer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DWaveConfig:
    """Configuration parameters for D-Wave solver."""
    api_token: str
    solver_name: Optional[str] = None
    num_reads: int = 1000
    annealing_time: int = 20  # microseconds
    chain_strength: Optional[float] = None
    auto_scale: bool = True
    use_fixed_embedding: bool = False
    topology: str = "zephyr"  # pegasus, zephyr, or chimera


@dataclass
class QuantumSolution:
    """Container for quantum annealing solution and metadata."""
    best_sample: Dict[str, int]
    energy: float
    num_occurrences: int
    is_feasible: bool
    selected_indices: np.ndarray
    timing_info: Dict[str, float]
    embedding_info: Dict[str, Any]
    all_samples: Optional[List[Dict]] = None


class DWaveClusteringSolver:
    """
    A solver class for clustering problems using D-Wave quantum annealing.
    
    This class manages the complete workflow of solving clustering problems
    on quantum hardware, including embedding, parameter tuning, and result
    interpretation.
    """
    
    def __init__(self, config: DWaveConfig):
        """
        Initialize the D-Wave solver with configuration.
        
        Args:
            config (DWaveConfig): Configuration object with D-Wave parameters
            
        Raises:
            ConnectionError: If unable to connect to D-Wave cloud service
        """
        self.config = config
        self.sampler = None
        self.composite_sampler = None
        self.fixed_embedding = None
        
        # Initialize connection to D-Wave
        self._initialize_sampler()
        
        logger.info(f"D-Wave solver initialized with {self.sampler.properties['chip_id']}")
        
    def _initialize_sampler(self) -> None:
        """
        Initialize the D-Wave sampler and embedding composite.
        
        This method establishes connection to the quantum hardware and
        sets up the embedding strategy (fixed or dynamic).
        """
        try:
            # Connect to D-Wave quantum annealer
            if self.config.solver_name:
                self.sampler = DWaveSampler(
                    token=self.config.api_token,
                    solver=self.config.solver_name
                )
            else:
                # Auto-select solver based on topology preference
                solver_features = {'topology__type': self.config.topology}
                self.sampler = DWaveSampler(
                    token=self.config.api_token,
                    solver=solver_features
                )
            
            # Set up embedding composite
            if self.config.use_fixed_embedding:
                # For repeated problems, use fixed embedding for consistency
                self.composite_sampler = FixedEmbeddingComposite(self.sampler, {})
            else:
                # Dynamic embedding for each problem
                self.composite_sampler = EmbeddingComposite(self.sampler)
                
            # Log hardware properties
            properties = self.sampler.properties
            logger.info(f"Connected to: {properties['chip_id']}")
            logger.info(f"Topology: {properties['topology']['type']}")
            logger.info(f"Qubits: {properties['num_qubits']}")
            logger.info(f"Couplers: {len(properties['couplers'])}")
            
        except Exception as e:
            logger.error(f"Failed to initialize D-Wave sampler: {e}")
            raise ConnectionError(f"Cannot connect to D-Wave: {e}")
    
    def solve_cluster_expansion(
        self,
        distance_matrix: np.ndarray,
        seed_cluster: np.ndarray,
        n_expand: int,
        lambda_penalty: float,
        candidate_points: Optional[np.ndarray] = None
    ) -> QuantumSolution:
        """
        Solve the cluster expansion problem using quantum annealing.
        
        This method takes a seed cluster and expands it by selecting exactly
        n_expand additional points that minimize the total intra-cluster distance.
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            seed_cluster (np.ndarray): Indices of points in the seed cluster
            n_expand (int): Number of points to add to the cluster
            lambda_penalty (float): Penalty parameter for cardinality constraint
            candidate_points (np.ndarray, optional): Candidate points for expansion
        
        Returns:
            QuantumSolution: Solution object containing results and metadata
            
        Example:
            >>> solver = DWaveClusteringSolver(config)
            >>> solution = solver.solve_cluster_expansion(
            ...     distance_matrix=dist_mat,
            ...     seed_cluster=np.array([0, 1, 2]),
            ...     n_expand=5,
            ...     lambda_penalty=2.0
            ... )
            >>> print(f"Selected points: {solution.selected_indices}")
        """
        logger.info(f"Solving cluster expansion: seed size={len(seed_cluster)}, "
                   f"expand by={n_expand}, λ={lambda_penalty:.3f}")
        
        # Create QUBO formulation
        from ..optimization.qubo_formulation import QUBOFormulator
        formulator = QUBOFormulator(distance_matrix)
        
        bqm = formulator.create_cluster_expansion_qubo(
            seed_cluster=seed_cluster,
            target_size=n_expand,
            lambda_penalty=lambda_penalty,
            candidate_points=candidate_points
        )
        
        # Solve on quantum hardware
        solution = self._solve_bqm(bqm, n_expand)
        
        # Post-process to get actual point indices
        if candidate_points is not None:
            selected_vars = [var for var, val in solution.best_sample.items() if val == 1]
            selected_indices = [candidate_points[int(var.split('_')[1])] for var in selected_vars]
            solution.selected_indices = np.array(selected_indices)
        
        return solution
    
    def _solve_bqm(
        self,
        bqm: BinaryQuadraticModel,
        target_size: int
    ) -> QuantumSolution:
        """
        Solve a Binary Quadratic Model on D-Wave hardware.
        
        This internal method handles the actual quantum annealing process,
        including chain strength calculation, embedding, and result extraction.
        
        Args:
            bqm (BinaryQuadraticModel): The QUBO model to solve
            target_size (int): Expected number of selected variables
        
        Returns:
            QuantumSolution: Processed solution with metadata
        """
        # Calculate chain strength if not provided
        if self.config.chain_strength is None:
            chain_strength = self._calculate_chain_strength(bqm)
        else:
            chain_strength = self.config.chain_strength
        
        logger.info(f"Using chain strength: {chain_strength:.3f}")
        
        # Prepare sampling parameters
        sample_params = {
            'num_reads': self.config.num_reads,
            'annealing_time': self.config.annealing_time,
            'chain_strength': chain_strength,
            'return_embedding': True,
            'answer_mode': 'histogram'  # Get aggregated results
        }
        
        # Execute quantum annealing
        start_time = time.time()
        try:
            sampleset = self.composite_sampler.sample(bqm, **sample_params)
        except Exception as e:
            logger.error(f"Quantum annealing failed: {e}")
            raise RuntimeError(f"D-Wave execution error: {e}")
        
        end_time = time.time()
        
        # Extract timing information
        timing_info = self._extract_timing_info(sampleset)
        timing_info['total_time'] = end_time - start_time
        
        # Extract embedding information
        embedding_info = self._extract_embedding_info(sampleset)
        
        # Get best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        best_occurrences = sampleset.first.num_occurrences
        
        # Validate solution feasibility
        selected_count = sum(best_sample.values())
        is_feasible = (selected_count == target_size)
        
        # Extract selected indices
        selected_vars = [var for var, val in best_sample.items() if val == 1]
        selected_indices = np.array([int(var.split('_')[1]) for var in selected_vars])
        
        # Log solution quality
        logger.info(f"Best solution: energy={best_energy:.3f}, "
                   f"occurrences={best_occurrences}/{self.config.num_reads}, "
                   f"feasible={is_feasible}")
        
        # Compile all samples if requested
        all_samples = None
        if hasattr(sampleset, 'record'):
            all_samples = [
                {
                    'sample': dict(zip(sampleset.variables, sample)),
                    'energy': energy,
                    'num_occurrences': num_occ
                }
                for sample, energy, num_occ in sampleset.record
            ]
        
        return QuantumSolution(
            best_sample=best_sample,
            energy=best_energy,
            num_occurrences=best_occurrences,
            is_feasible=is_feasible,
            selected_indices=selected_indices,
            timing_info=timing_info,
            embedding_info=embedding_info,
            all_samples=all_samples
        )
    
    def _calculate_chain_strength(self, bqm: BinaryQuadraticModel) -> float:
        """
        Calculate appropriate chain strength for the problem.
        
        Chain strength needs to be large enough to maintain chain integrity
        but not so large that it dominates the problem energy scale.
        
        Args:
            bqm (BinaryQuadraticModel): The QUBO model
        
        Returns:
            float: Calculated chain strength
        """
        # Get the range of coefficients in the BQM
        linear_range = 0
        if bqm.linear:
            linear_vals = list(bqm.linear.values())
            linear_range = max(linear_vals) - min(linear_vals)
        
        quadratic_range = 0
        if bqm.quadratic:
            quadratic_vals = list(bqm.quadratic.values())
            quadratic_range = max(quadratic_vals) - min(quadratic_vals)
        
        # Chain strength should be larger than the problem energy scale
        # Common heuristic: 1.5-2x the maximum coefficient magnitude
        max_coeff = max(linear_range, quadratic_range)
        
        # Add some buffer for constraint terms (which are typically larger)
        chain_strength = 1.5 * max_coeff
        
        # Ensure minimum chain strength
        chain_strength = max(chain_strength, 1.0)
        
        logger.info(f"Calculated chain strength: {chain_strength:.3f} "
                   f"(linear range: {linear_range:.3f}, "
                   f"quadratic range: {quadratic_range:.3f})")
        
        return chain_strength
    
    def _extract_timing_info(self, sampleset) -> Dict[str, float]:
        """
        Extract detailed timing information from the sampleset.
        
        Args:
            sampleset: D-Wave sampleset object
        
        Returns:
            Dict[str, float]: Timing breakdown in seconds
        """
        timing = {}
        
        if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
            dwave_timing = sampleset.info['timing']
            
            # Convert microseconds to seconds for relevant fields
            timing_fields = {
                'qpu_access_time': 1e-6,  # Total QPU access time
                'qpu_programming_time': 1e-6,  # Time to program the QPU
                'qpu_sampling_time': 1e-6,  # Actual annealing time
                'qpu_anneal_time_per_sample': 1e-6,  # Annealing time per sample
                'qpu_readout_time_per_sample': 1e-6,  # Readout time per sample
                'qpu_delay_time_per_sample': 1e-6,  # Delay between samples
                'post_processing_overhead_time': 1e-6,  # Post-processing time
                'total_post_processing_time': 1e-6  # Total post-processing
            }
            
            for field, conversion in timing_fields.items():
                if field in dwave_timing:
                    timing[field] = dwave_timing[field] * conversion
            
            # Log timing breakdown
            logger.debug("QPU Timing breakdown:")
            for key, value in timing.items():
                logger.debug(f"  {key}: {value:.6f} seconds")
        
        return timing
    
    def _extract_embedding_info(self, sampleset) -> Dict[str, Any]:
        """
        Extract embedding information from the sampleset.
        
        Args:
            sampleset: D-Wave sampleset object
        
        Returns:
            Dict[str, Any]: Embedding statistics
        """
        embedding_info = {}
        
        if hasattr(sampleset, 'info') and 'embedding_context' in sampleset.info:
            context = sampleset.info['embedding_context']
            
            if 'embedding' in context:
                embedding = context['embedding']
                
                # Calculate embedding statistics
                chain_lengths = [len(chain) for chain in embedding.values()]
                
                embedding_info = {
                    'num_logical_variables': len(embedding),
                    'num_physical_qubits': sum(chain_lengths),
                    'max_chain_length': max(chain_lengths) if chain_lengths else 0,
                    'avg_chain_length': np.mean(chain_lengths) if chain_lengths else 0,
                    'min_chain_length': min(chain_lengths) if chain_lengths else 0
                }
                
                # Check for chain breaks
                if 'chain_break_fraction' in context:
                    embedding_info['chain_break_fraction'] = context['chain_break_fraction']
                
                logger.debug(f"Embedding stats: {embedding_info}")
        
        return embedding_info
    
    def batch_solve(
        self,
        problems: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[QuantumSolution]:
        """
        Solve multiple clustering problems in batch.
        
        This method can optionally reuse embeddings for similar problems
        to improve performance.
        
        Args:
            problems (List[Dict]): List of problem specifications
            parallel (bool): Whether to use parallel embedding (if available)
        
        Returns:
            List[QuantumSolution]: Solutions for all problems
        """
        solutions = []
        
        # If using fixed embedding, compute it once for the first problem
        if self.config.use_fixed_embedding and problems:
            first_problem = problems[0]
            # Generate embedding from first problem
            # ... (embedding generation code)
        
        for i, problem in enumerate(problems):
            logger.info(f"Solving problem {i+1}/{len(problems)}")
            
            solution = self.solve_cluster_expansion(
                distance_matrix=problem['distance_matrix'],
                seed_cluster=problem.get('seed_cluster', np.array([])),
                n_expand=problem['n_expand'],
                lambda_penalty=problem['lambda_penalty']
            )
            
            solutions.append(solution)
        
        return solutions
    
    def optimize_annealing_schedule(
        self,
        bqm: BinaryQuadraticModel,
        test_schedules: Optional[List[List[Tuple[float, float]]]] = None
    ) -> Dict[str, Any]:
        """
        Experiment with different annealing schedules to find optimal parameters.
        
        Args:
            bqm (BinaryQuadraticModel): Problem to test
            test_schedules (List): Custom annealing schedules to test
        
        Returns:
            Dict[str, Any]: Best schedule and performance metrics
        """
        if test_schedules is None:
            # Default schedules to test
            test_schedules = [
                [(0.0, 0.0), (20.0, 1.0)],  # Standard linear
                [(0.0, 0.0), (10.0, 0.5), (20.0, 1.0)],  # Pause in middle
                [(0.0, 0.0), (5.0, 0.8), (20.0, 1.0)],  # Fast ramp
            ]
        
        best_schedule = None
        best_energy = float('inf')
        results = []
        
        for schedule in test_schedules:
            logger.info(f"Testing schedule: {schedule}")
            
            # Sample with custom schedule
            sampleset = self.sampler.sample(
                bqm,
                num_reads=100,
                anneal_schedule=schedule
            )
            
            # Record results
            mean_energy = np.mean([s.energy for s in sampleset])
            min_energy = sampleset.first.energy
            
            results.append({
                'schedule': schedule,
                'mean_energy': mean_energy,
                'min_energy': min_energy
            })
            
            if min_energy < best_energy:
                best_energy = min_energy
                best_schedule = schedule
        
        return {
            'best_schedule': best_schedule,
            'best_energy': best_energy,
            'all_results': results
        }
