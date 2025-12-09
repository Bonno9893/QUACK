#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Approximate Optimization Algorithm (QAOA) for Binary Clustering
=======================================================================

This module implements a comprehensive benchmarking framework for QAOA applied to 
the binary clustering problem (K=2). It includes performance metrics for both 
quantum and classical computational components, supports multiple quantum backends,
and provides detailed analysis of the optimization landscape.

Key Features:
- Multi-backend support (GPU/CPU quantum simulators)
- Comprehensive performance metrics (quantum vs classical time separation)
- Resource monitoring (CPU/GPU utilization)
- Automatic result streaming for real-time monitoring
- Early stopping mechanisms for efficient computation
- Detailed histogram visualization of quantum state distributions

References:
- Farhi, E., et al. "A quantum approximate optimization algorithm." 
  arXiv preprint arXiv:1411.4028 (2014).
- Zhou, L., et al. "Quantum approximate optimization algorithm: Performance, 
  mechanism, and implementation on near-term devices." 
  Physical Review X 10.2 (2020): 021067.

Author: [Your Research Group]
License: MIT
"""

import os
import pickle
import time
import argparse
import json
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import cudaq
from cudaq import spin
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import threading
import platform
import subprocess
import scipy.optimize
from scipy.optimize import minimize
from contextlib import contextmanager
from functools import wraps
from scipy.stats import entropy as scipy_entropy

# ============================================================================
# External Library Availability Checks
# ============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("WARNING: psutil not installed. CPU metrics will not be available.")
    print("Install with: pip install psutil")
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    print("WARNING: pynvml not available. GPU metrics will not be available.")
    print("Install with: pip install nvidia-ml-py")
    PYNVML_AVAILABLE = False

# ============================================================================
# Quantum Operation Timing Infrastructure
# ============================================================================

class CudaqTimerStore:
    """
    Global store for accumulating execution times of CUDA-Q quantum operations.
    
    This class tracks the cumulative time spent in quantum operations
    (observe and sample) to separate quantum from classical computation time.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing counters to zero."""
        self.t_observe = 0.0  # Total time in observe operations
        self.t_sample = 0.0   # Total time in sample operations
        self.n_observe = 0    # Number of observe calls
        self.n_sample = 0     # Number of sample calls

# Global timer store instance
timer_store = CudaqTimerStore()

def timed_observe(original_observe):
    """
    Decorator to measure execution time of cudaq.observe operations.
    
    Args:
        original_observe: The original cudaq.observe function
        
    Returns:
        Wrapped function that tracks timing
    """
    @wraps(original_observe)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = original_observe(*args, **kwargs)
        dt = time.perf_counter() - start
        timer_store.t_observe += dt
        timer_store.n_observe += 1
        return result
    return wrapper

def timed_sample(original_sample):
    """
    Decorator to measure execution time of cudaq.sample operations.
    
    Args:
        original_sample: The original cudaq.sample function
        
    Returns:
        Wrapped function that tracks timing
    """
    @wraps(original_sample)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = original_sample(*args, **kwargs)
        dt = time.perf_counter() - start
        timer_store.t_sample += dt
        timer_store.n_sample += 1
        return result
    return wrapper

@contextmanager
def patch_cudaq_timers():
    """
    Context manager to temporarily patch CUDA-Q functions for timing measurements.
    
    This allows us to separate quantum computation time from classical optimization
    overhead within the VQE algorithm.
    
    Yields:
        CudaqTimerStore: The timer store with accumulated timings
    """
    global timer_store
    timer_store.reset()
    
    # Save original functions
    _obs = cudaq.observe
    _smp = cudaq.sample
    
    # Apply timing decorators
    cudaq.observe = timed_observe(_obs)
    cudaq.sample = timed_sample(_smp)
    
    try:
        yield timer_store
    finally:
        # Restore original functions
        cudaq.observe = _obs
        cudaq.sample = _smp

# ============================================================================
# System Resource Monitoring
# ============================================================================

def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Retrieve GPU information if available.
    
    Returns:
        Dictionary containing GPU name and driver version, or None if unavailable
    """
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None
        
        # Use first GPU device
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = {
            'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
            'driver': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
        }
        return gpu_info
    except:
        return None

def monitor_system(stop_event: threading.Event, data_list: List[Dict], 
                   interval: float = 0.25, pid: Optional[int] = None):
    """
    Thread function for monitoring CPU and GPU usage during execution.
    
    Args:
        stop_event: Threading event to signal monitoring stop
        data_list: List to append monitoring data
        interval: Sampling interval in seconds
        pid: Process ID to monitor (defaults to current process)
    """
    if pid is None:
        pid = os.getpid()
    
    process = psutil.Process(pid) if PSUTIL_AVAILABLE else None
    gpu_handle = None
    
    if PYNVML_AVAILABLE:
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pass
    
    while not stop_event.is_set():
        data = {'timestamp': time.time()}
        
        # CPU metrics
        if process:
            try:
                data['cpu_percent'] = process.cpu_percent(interval=0.1)
                data['rss_bytes'] = process.memory_info().rss
            except:
                pass
        
        # GPU metrics
        if gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                data['gpu_util'] = util.gpu
                data['gpu_mem_util'] = util.memory
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                data['gpu_mem_used'] = mem_info.used
                
                # Optional: power and temperature
                try:
                    data['gpu_power'] = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # W
                except:
                    pass
                
                try:
                    data['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(
                        gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    pass
            except:
                pass
        
        data_list.append(data)
        time.sleep(interval)

def start_sys_monitor(interval: float = 0.25) -> Tuple[threading.Event, List[Dict]]:
    """
    Start system monitoring in a separate thread.
    
    Args:
        interval: Monitoring interval in seconds
        
    Returns:
        Tuple of (stop_event, data_list) for controlling and accessing monitor
    """
    stop_event = threading.Event()
    data_list = []
    
    thread = threading.Thread(
        target=monitor_system, 
        args=(stop_event, data_list, interval)
    )
    thread.daemon = True
    thread.start()
    
    return stop_event, data_list

def stop_sys_monitor(stop_event: threading.Event, data_list: List[Dict]) -> Dict[str, float]:
    """
    Stop system monitoring and return aggregated metrics.
    
    Args:
        stop_event: Threading event to signal stop
        data_list: List containing monitoring data
        
    Returns:
        Dictionary with aggregated metrics (mean, max, min for each metric)
    """
    stop_event.set()
    time.sleep(0.5)  # Allow thread to finish
    
    if not data_list:
        return {}
    
    # Aggregate metrics
    df = pd.DataFrame(data_list)
    metrics = {}
    
    for col in df.columns:
        if col == 'timestamp':
            continue
        if col in df:
            metrics[f'{col}_mean'] = df[col].mean()
            metrics[f'{col}_max'] = df[col].max()
            metrics[f'{col}_min'] = df[col].min()
    
    return metrics

# ============================================================================
# QAOA Implementation for Binary Clustering
# ============================================================================

def hamiltonian_k2(distance_matrix: np.ndarray, scale: Optional[float] = None) -> Any:
    """
    Construct the QAOA cost Hamiltonian for binary clustering.
    
    The Hamiltonian minimizes intra-cluster distances:
    H = Σ_{i<j} d_ij / 2 * (I + Z_i Z_j)
    
    Args:
        distance_matrix: Symmetric matrix of pairwise distances
        scale: Optional scaling factor for distances (defaults to max distance)
        
    Returns:
        CUDA-Q spin operator representing the Hamiltonian
    """
    n = len(distance_matrix)
    H = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i][j] if scale is None else distance_matrix[i][j] / scale
            H += 0.5 * d * (spin.i(i) * spin.i(j)   # Identity term
                            + spin.z(i) * spin.z(j)) # Interaction term
    return H

def calculate_clustering_cost(distance_matrix: np.ndarray, bitstring: str) -> float:
    """
    Calculate the clustering cost for a given bit string assignment.
    
    The cost is the sum of distances between points in the same cluster.
    
    Args:
        distance_matrix: Symmetric matrix of pairwise distances
        bitstring: Binary string representing cluster assignments
        
    Returns:
        Total intra-cluster distance cost
    """
    n = len(bitstring)
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if bitstring[i] == bitstring[j]:
                cost += distance_matrix[i][j]
    return cost

@cudaq.kernel
def kernel_qaoa(n_qubits: int, p: int,
                edges_src: list[int], edges_tgt: list[int],
                weights: list[float], thetas: list[float]):
    """
    QAOA quantum kernel for binary clustering.
    
    Implements the standard QAOA ansatz:
    |ψ⟩ = ∏_{k=1}^p [U_B(β_k) U_C(γ_k)] |+⟩^⊗n
    
    where U_C is the cost unitary and U_B is the mixing unitary.
    
    Args:
        n_qubits: Number of qubits (equal to number of data points)
        p: Number of QAOA layers
        edges_src: Source vertices for edges
        edges_tgt: Target vertices for edges
        weights: Edge weights (normalized distances)
        thetas: Parameter vector [γ_1, β_1, γ_2, β_2, ..., γ_p, β_p]
    """
    q = cudaq.qvector(n_qubits)
    h(q)  # Initialize in uniform superposition
    
    for layer in range(p):
        gamma = thetas[2 * layer]
        beta = thetas[2 * layer + 1]
        
        # Cost unitary: exp(-i γ H_C)
        for e in range(len(edges_src)):
            u = edges_src[e]
            v = edges_tgt[e]
            w = weights[e]
            x.ctrl(q[u], q[v])
            rz(gamma * w, q[v])
            x.ctrl(q[u], q[v])
        
        # Mixing unitary: exp(-i β H_B)
        for i in range(n_qubits):
            rx(2.0 * beta, q[i])

# ============================================================================
# Backend Selection and Configuration
# ============================================================================

def pick_backend(preferred: str = "nvidia") -> str:
    """
    Select and configure the appropriate quantum simulation backend.
    
    Args:
        preferred: Preferred backend name (e.g., "nvidia", "qpp-cpu")
        
    Returns:
        Name of the selected backend
    """
    targets = {t.name for t in cudaq.get_targets()}
    print(f"[CUDA-Q] Available targets: {targets}")
    
    selected = None
    
    # Try preferred backend with fp64 precision
    if preferred in targets:
        try:
            cudaq.set_target(preferred, option="fp64")
            print(f"[CUDA-Q] Backend = {preferred} with fp64")
            selected = preferred
        except:
            try:
                cudaq.set_target(preferred)
                print(f"[CUDA-Q] Backend = {preferred}")
                selected = preferred
            except:
                pass
    
    # Fallback to CPU backends
    if selected is None:
        for name in ["qpp-cpu", "density-matrix-cpu"]:
            if name in targets:
                try:
                    cudaq.set_target(name)
                    print(f"[CUDA-Q] Backend = {name}")
                    selected = name
                    break
                except:
                    pass
    
    if selected is None:
        print("[CUDA-Q] Using default backend")
        selected = "default"
    
    return selected

def shots_for_n(n: int, base: int = 5000) -> int:
    """
    Heuristic for determining number of measurement shots based on problem size.
    
    The number of shots scales with the Hilbert space dimension to maintain
    adequate sampling coverage.
    
    Args:
        n: Number of qubits
        base: Base number of shots
        
    Returns:
        Recommended number of shots (capped at 20,000)
    """
    shots = int(base * np.sqrt(2**n) / 8)
    shots = max(shots, 1000)  # Minimum threshold
    return min(shots, 20000)   # Maximum cap

def make_shots_grid(n: int, base: int = 5000) -> List[int]:
    """
    Generate a grid of shot counts for systematic testing.
    
    Args:
        n: Number of qubits
        base: Base number of shots
        
    Returns:
        List of shot counts to test
    """
    base_shots = shots_for_n(n, base)
    grid = [
        max(100, base_shots // 4),
        max(200, base_shots // 2),
        base_shots,
        base_shots * 2,
        base_shots * 4
    ]
    return sorted(set(grid))

# ============================================================================
# QAOA Execution and Analysis
# ============================================================================

def run_qaoa(distance_matrix: np.ndarray, y_true: np.ndarray, n: int, p: int, 
             shots: int, seed: int, iterations: int = 4000, 
             theta0: Optional[np.ndarray] = None, track_metrics: bool = True, 
             skip_vqe: bool = False) -> Tuple[float, float, float, np.ndarray, 
                                               Dict, Dict, Dict]:
    """
    Execute a single QAOA run with comprehensive metric tracking.
    
    This function performs the complete QAOA workflow: VQE optimization followed
    by sampling and analysis of the resulting quantum state distribution.
    
    Args:
        distance_matrix: Symmetric matrix of pairwise distances
        y_true: Ground truth cluster labels
        n: Number of data points (qubits)
        p: Number of QAOA layers
        shots: Number of measurement shots
        seed: Random seed for reproducibility
        iterations: Maximum VQE iterations
        theta0: Initial or fixed parameters (warm start)
        track_metrics: Whether to track detailed timing metrics
        skip_vqe: Skip VQE optimization and use provided parameters
        
    Returns:
        Tuple containing:
        - best_cost: Lowest clustering cost found
        - prob_best: Probability of measuring the best solution
        - ari: Adjusted Rand Index compared to ground truth
        - theta_used: Optimized or used parameters
        - results_dict: Complete distribution of measurements
        - times: Detailed timing breakdown
        - metrics: Additional performance metrics
    """
    
    def _expand_theta_for_p(theta_in: Optional[np.ndarray], p_target: int) -> Optional[np.ndarray]:
        """Expand parameter vector from p-1 layers to p layers by repeating last pair."""
        if theta_in is None:
            return None
        theta_in = np.asarray(theta_in, dtype=float)
        dim_target = 2 * p_target
        if theta_in.size == dim_target:
            return theta_in
        if theta_in.size >= 2:
            # Repeat last (γ, β) pair
            last_pair = theta_in[-2:]
            k = (dim_target - theta_in.size) // 2
            return np.concatenate([theta_in, np.tile(last_pair, k)])
        return None
    
    # Initialize timing dictionary
    times = dict(
        t_total=0, t_vqe=0, t_classic=0,
        t_quantum=0, t_observe=0, t_sample=0,
        t_sampling=0,
        n_observe=0, n_sample=0
    )
    
    t0 = time.perf_counter()
    
    # Normalize distances for numerical stability
    max_d = float(np.max(distance_matrix)) if np.max(distance_matrix) > 0 else 1.0
    
    # Build Hamiltonian
    H = hamiltonian_k2(distance_matrix, scale=max_d)
    
    # Prepare edge data with normalized weights
    edges_src, edges_tgt, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            edges_src.append(i)
            edges_tgt.append(j)
            weights.append(distance_matrix[i][j] / max_d)
    
    # Parameter dimension
    dim = 2 * p
    
    # VQE Optimization Phase
    np.random.seed(seed)
    
    theta_used = None
    if skip_vqe:
        # Use provided parameters or random initialization
        theta_used = _expand_theta_for_p(theta0, p)
        if theta_used is None:
            theta_used = np.random.uniform(-np.pi, np.pi, dim)
    else:
        # Initialize parameters
        if theta0 is None:
            init = np.random.uniform(-np.pi, np.pi, dim)
        else:
            init = _expand_theta_for_p(theta0, p)
            if init is None:
                init = np.random.uniform(-np.pi, np.pi, dim)
        
        # Configure optimizer
        opt = cudaq.optimizers.NelderMead()
        opt.initial_parameters = init
        opt.max_iterations = iterations
        
        # Run VQE with optional timing
        t1 = time.perf_counter()
        if track_metrics:
            with patch_cudaq_timers() as mt:
                Eopt, theta_used = cudaq.vqe(
                    kernel=kernel_qaoa,
                    spin_operator=H,
                    argument_mapper=lambda t: (n, p, edges_src, edges_tgt, weights, t),
                    parameter_count=dim,
                    optimizer=opt
                )
            times['t_observe'] = mt.t_observe
            times['n_observe'] = mt.n_observe
        else:
            Eopt, theta_used = cudaq.vqe(
                kernel=kernel_qaoa,
                spin_operator=H,
                argument_mapper=lambda t: (n, p, edges_src, edges_tgt, weights, t),
                parameter_count=dim,
                optimizer=opt
            )
        times['t_vqe'] = time.perf_counter() - t1
    
    # Sampling Phase
    t2 = time.perf_counter()
    cudaq.set_random_seed(seed)
    if track_metrics:
        with patch_cudaq_timers() as mt:
            counts = cudaq.sample(kernel_qaoa, n, p, edges_src, edges_tgt, weights,
                                  theta_used, shots_count=shots)
        times['t_sample'] = mt.t_sample
        times['n_sample'] = mt.n_sample
    else:
        counts = cudaq.sample(kernel_qaoa, n, p, edges_src, edges_tgt, weights,
                              theta_used, shots_count=shots)
        times['n_sample'] = 1
    times['t_sampling'] = time.perf_counter() - t2
    
    # Calculate quantum vs classical time breakdown
    times['t_quantum'] = times['t_observe'] + times['t_sample']
    times['t_classic'] = times['t_vqe'] - times['t_observe']
    times['t_total'] = time.perf_counter() - t0
    
    # Analyze results distribution
    results_dict = {}
    for bs, cnt in counts.items():
        cst = calculate_clustering_cost(distance_matrix, bs)
        results_dict[bs] = (cnt, cst, cst)  # (count, energy, cost)
    
    # Find best solution
    best_bs = min(results_dict.keys(), key=lambda b: results_dict[b][2])
    best_cost = results_dict[best_bs][2]
    
    # Calculate probability of finding optimal solution
    prob_best = sum(c for b, (c, _, k) in results_dict.items()
                    if abs(k - best_cost) < 1e-9) / max(shots, 1)
    
    # Evaluate clustering quality
    clustering = [int(b) for b in best_bs]
    ari = adjusted_rand_score(y_true, clustering)
    
    # Additional metrics
    metrics = {
        'n_unique_bitstrings': len(results_dict)
    }
    
    return best_cost, prob_best, ari, theta_used, results_dict, times, metrics

# ============================================================================
# Performance Analysis Utilities
# ============================================================================

def calculate_additional_metrics(results_dict: Dict, optimal_cost: float, 
                                 shots: int, tol: float = 1e-9) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from sampling results.
    
    Args:
        results_dict: Dictionary mapping bitstrings to (count, energy, cost)
        optimal_cost: Known optimal clustering cost
        shots: Total number of measurement shots
        tol: Tolerance for comparing costs
        
    Returns:
        Dictionary containing:
        - prob_opt: Probability of finding exact optimal solution
        - prob_valid: Probability of finding solution within 5% of optimal
        - entropy_samples: Shannon entropy of the distribution
        - n_opt_states: Number of optimal states observed
        - max_non_opt_prob: Maximum probability among non-optimal states
        - top_non_opt_bs: Bitstring of most probable non-optimal state
    """
    # Probability of finding exactly optimal solution
    prob_opt = sum(cnt for bs, (cnt, e, c) in results_dict.items()
                   if abs(c - optimal_cost) < tol) / max(shots, 1)
    
    # Probability of finding solution within 5% of optimal
    threshold = optimal_cost * 1.05 if optimal_cost > 0 else 0.05
    prob_valid = sum(cnt for bs, (cnt, e, c) in results_dict.items()
                     if c <= threshold) / max(shots, 1)
    
    # Distribution entropy
    counts = [cnt for cnt, _, _ in results_dict.values()]
    if sum(counts) > 0:
        probs = np.array(counts) / sum(counts)
        entropy_samples = scipy_entropy(probs, base=2)
    else:
        entropy_samples = 0.0
    
    # Count of optimal configurations observed
    n_opt_states = sum(1 for bs, (cnt, e, c) in results_dict.items()
                       if cnt > 0 and abs(c - optimal_cost) < tol)
    
    # Maximum probability among non-optimal states
    max_non_opt_prob = 0.0
    top_non_opt_bs = None
    for bs, (cnt, e, c) in results_dict.items():
        if c > optimal_cost + tol:
            p = cnt / max(shots, 1)
            if p > max_non_opt_prob:
                max_non_opt_prob = p
                top_non_opt_bs = bs
    
    return {
        'prob_opt': prob_opt,
        'prob_valid': prob_valid,
        'entropy_samples': entropy_samples,
        'n_opt_states': n_opt_states,
        'max_non_opt_prob': max_non_opt_prob,
        'top_non_opt_bs': top_non_opt_bs
    }

def get_system_info() -> Dict[str, Any]:
    """
    Collect comprehensive system and hardware information.
    
    Returns:
        Dictionary containing system specifications, software versions,
        and available hardware accelerators
    """
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'cpu_model': platform.processor(),
        'cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 'N/A',
        'ram_total_gb': psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 'N/A',
    }
    
    # GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        info.update({f'gpu_{k}': v for k, v in gpu_info.items()})
    
    # Software versions
    try:
        info['python_version'] = platform.python_version()
        info['numpy_version'] = np.__version__
        info['cudaq_version'] = cudaq.__version__ if hasattr(cudaq, '__version__') else 'unknown'
    except:
        pass
    
    # Git commit if in repository
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        info['git_commit'] = commit[:8]
    except:
        pass
    
    return info

# ============================================================================
# Main Benchmark Framework
# ============================================================================

class QAOAClusteringBenchmark:
    """
    Comprehensive benchmarking framework for QAOA on clustering instances.
    
    This class orchestrates the execution of QAOA experiments across multiple
    problem instances, parameter settings, and quantum backends. It provides
    automated result collection, performance analysis, and visualization.
    
    Attributes:
        istanze_dir: Directory containing problem instance files
        results_dir: Directory for output files
        backend_name: Currently selected quantum backend
        system_info: System hardware and software information
    """
    
    def __init__(self, istanze_dir: str, results_dir: str = "qaoa_k2_results"):
        """
        Initialize the benchmark framework.
        
        Args:
            istanze_dir: Path to directory containing instance pickle files
            results_dir: Path to directory for saving results
        """
        self.istanze_dir = istanze_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.backend_name = None
        self.system_info = get_system_info()
        
        # Display system information
        print("\nSystem Information:")
        for k, v in self.system_info.items():
            if k != 'backend':
                print(f"  {k}: {v}")
        print()
    
    def load_instance(self, path: str) -> Dict:
        """Load a problem instance from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def find_optimal_solution(self, distance_matrix: np.ndarray, 
                              n_points: int) -> Tuple[str, float]:
        """
        Find the optimal clustering solution via exhaustive search.
        
        Args:
            distance_matrix: Symmetric matrix of pairwise distances
            n_points: Number of data points
            
        Returns:
            Tuple of (optimal_bitstring, optimal_cost)
        """
        best_cost = float('inf')
        best_solution = None
        
        for i in range(2**n_points):
            bitstring = format(i, f'0{n_points}b')
            cost = calculate_clustering_cost(distance_matrix, bitstring)
            if cost < best_cost:
                best_cost = cost
                best_solution = bitstring
        
        return best_solution, best_cost
    
    def plot_test_histogram(self, results_dict: Dict, instance_name: str, 
                            n_points: int, layers: int, shots: int, 
                            optimal_cost: float, save_dir: str, 
                            hist_min_pct: float = 2.0) -> float:
        """
        Create histogram visualization of quantum state distribution.
        
        The histogram shows both the frequency of measurement outcomes and their
        associated clustering costs. Only significant states are displayed based
        on filtering criteria.
        
        Args:
            results_dict: Dictionary mapping bitstrings to (count, energy, cost)
            instance_name: Name of the problem instance
            n_points: Number of data points
            layers: Number of QAOA layers
            shots: Number of measurement shots
            optimal_cost: Known optimal clustering cost
            save_dir: Directory to save the histogram
            hist_min_pct: Minimum percentage threshold for display
            
        Returns:
            Time taken to generate the histogram
        """
        t_hist_start = time.perf_counter()
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Prepare data
        bitstrings = list(results_dict.keys())
        counts = [results_dict[bs][0] for bs in bitstrings]
        costs = [results_dict[bs][2] for bs in bitstrings]
        
        # Filter states for display
        counts = np.asarray(counts, dtype=int)
        costs = np.asarray(costs, dtype=float)
        
        thr_count = max(1, int(np.ceil(hist_min_pct / 100.0 * shots)))
        best_cost_obs = float(costs.min()) if len(costs) else float('inf')
        valid_thr = (optimal_cost * 1.05) if optimal_cost > 0 else (best_cost_obs * 1.05)
        
        # Keep: frequent states, optimal states, or near-optimal states
        keep_idx = [
            i for i, (cnt, cst) in enumerate(zip(counts, costs))
            if (cnt >= thr_count) or (abs(cst - optimal_cost) < 1e-9) or (cst <= valid_thr)
        ]
        
        if not keep_idx:
            # Fallback: keep top-N by count
            keep_idx = np.argsort(counts)[-min(20, len(counts)):].tolist()
        
        # Apply filter
        bitstrings = [bitstrings[i] for i in keep_idx]
        counts = counts[keep_idx]
        costs = costs[keep_idx]
        
        # Sort by cost
        order = np.argsort(costs)
        bitstrings = [bitstrings[i] for i in order]
        counts = counts[order]
        costs = costs[order]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Subplot 1: Frequency
        bars1 = ax1.bar(range(len(bitstrings)), counts, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Sampling Frequency', fontsize=12)
        ax1.set_title(f'{instance_name} - {n_points} points, {layers} layers, {shots} shots', 
                      fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight frequent solutions
        label_thr = thr_count
        max_count = max(counts) if len(counts) > 0 else 1
        for i, (count, bs) in enumerate(zip(counts, bitstrings)):
            if count >= label_thr:
                percentage = count / shots * 100
                ax1.text(i, count + max_count*0.01, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
            if count == max_count:
                bars1[i].set_color('darkblue')
                bars1[i].set_alpha(1.0)
        
        # Subplot 2: Cost
        colors = []
        for cost in costs:
            if abs(cost - optimal_cost) < 0.01:
                colors.append('darkgreen')
            elif abs(cost - optimal_cost) < max(0.1 * optimal_cost, 1e-6):
                colors.append('forestgreen')
            else:
                colors.append('orange')
        
        bars2 = ax2.bar(range(len(bitstrings)), costs, color=colors, alpha=0.8)
        ax2.set_xlabel('Bitstring', fontsize=12)
        ax2.set_ylabel('Clustering Cost', fontsize=12)
        ax2.set_xticks(range(len(bitstrings)))
        ax2.set_xticklabels(bitstrings, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Optimal cost line
        ax2.axhline(y=optimal_cost, color='red', linestyle='--', alpha=0.7, 
                    label=f'Optimal cost: {optimal_cost:.2f}')
        ax2.legend()
        
        # Add cost values
        for i, cost in enumerate(costs):
            if counts[i] >= thr_count:
                cost_range = (max(costs) - min(costs)) if len(costs) > 1 else 1
                ax2.text(i, cost + cost_range*0.01, 
                        f'{cost:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        filename = f"{instance_name}_n{n_points}_l{layers}_s{shots}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        t_hist = time.perf_counter() - t_hist_start
        print(f"    Histogram saved: {filename} (time: {t_hist:.2f}s)")
        
        return t_hist
    
    def run_benchmark(self, config: Dict) -> pd.DataFrame:
        """
        Execute the complete benchmark suite.
        
        This method orchestrates the execution of QAOA experiments across all
        specified configurations, collecting comprehensive performance metrics
        and generating analysis reports.
        
        Args:
            config: Configuration dictionary specifying:
                - layers: List of QAOA depths to test
                - shots_base: Base number of measurement shots
                - seeds: Random seeds for reproducibility
                - instances: Specific instances to test (None for all)
                - max_iterations: Maximum VQE iterations
                - track_metrics: Enable detailed metric tracking
                - save_histograms: Generate histogram visualizations
                - monitor_interval: System monitoring interval
                - backends: List of quantum backends to test
                - hist_min_pct: Histogram filtering threshold
                - early_stop: Enable early stopping
                - es_* : Early stopping parameters
                - stream_results: Enable real-time result streaming
                
        Returns:
            DataFrame containing all experimental results
        """
        # Default configuration
        default_config = {
            "layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "shots_base": 1000,
            "seeds": [27],
            "instances": None,
            "max_iterations": 10000,
            "track_metrics": True,
            "save_histograms": True,
            "monitor_interval": 0.25,
            "backends": ["nvidia", "qpp-cpu"],
            "hist_min_pct": 2.0,
            "early_stop": True,
            "es_P_dom": 0.10,
            "es_Q_others": 0.02,
            "es_deltaP_min": 0.02,
            "es_patience_layers": 2,
            "es_stop_shots_if_dominant": True,
            "stream_results": True
        }
        
        # Merge with provided configuration
        cfg = {**default_config, **config}
        
        # Load instance list
        selected_file = 'istanze_k2_selected.txt'
        if not os.path.exists(selected_file):
            print(f"Warning: {selected_file} not found. Using all instances in directory.")
            all_instances = [f for f in os.listdir(self.istanze_dir) if f.endswith('.pkl')]
        else:
            with open(selected_file, 'r') as f:
                all_instances = [line.strip() for line in f if line.strip()]
        
        # Filter instances if specified
        if cfg["instances"]:
            instances = [i for i in all_instances if i in cfg["instances"]]
        else:
            instances = all_instances
        
        # Display configuration
        print(f"\n{'='*80}")
        print(f"QAOA CLUSTERING K=2 BENCHMARK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Backends to test: {cfg['backends']}")
        print(f"Instances to process: {len(instances)}")
        print(f"Layers: {cfg['layers']}")
        print(f"Seeds: {cfg['seeds']}")
        print(f"Shots base: {cfg['shots_base']}")
        print(f"Max iterations: {cfg['max_iterations']}")
        print(f"Track metrics: {cfg['track_metrics']}")
        print(f"Histogram min %: {cfg['hist_min_pct']}%")
        print(f"Early stopping: {cfg['early_stop']} (P_dom={cfg['es_P_dom']}, "
              f"Q_others={cfg['es_Q_others']}, patience={cfg['es_patience_layers']})")
        print(f"Stream results: {cfg['stream_results']}")
        print(f"{'='*80}")
        
        all_results = []
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        stream_paths = {}
        
        def append_stream_row(backend_name: str, row: dict):
            """Append a single row to streaming CSV for real-time monitoring."""
            if not cfg.get("stream_results", True):
                return None
            stream_path = stream_paths.get(backend_name)
            if stream_path is None:
                stream_path = os.path.join(self.results_dir, f"stream_{backend_name}_{run_id}.csv")
                stream_paths[backend_name] = stream_path
            df_row = pd.DataFrame([row])
            header_needed = not os.path.exists(stream_path)
            df_row.to_csv(stream_path, mode='a', index=False, header=header_needed)
            return stream_path
        
        # Loop over backends
        for backend in cfg["backends"]:
            print(f"\n{'='*60}")
            print(f"TESTING BACKEND: {backend}")
            print(f"{'='*60}")
            
            # Setup backend
            try:
                self.backend_name = pick_backend(backend)
            except Exception as e:
                print(f"ERROR: Cannot select backend {backend}: {e}")
                continue
            
            if cfg.get("stream_results", True):
                preview_path = os.path.join(self.results_dir, f"stream_{self.backend_name}_{run_id}.csv")
                print(f"[STREAM] Writing incremental results to: {preview_path}")
            
            # Process each instance
            for idx, instance_file in enumerate(instances):
                print(f"\n[{backend}][{idx+1}/{len(instances)}] Processing {instance_file}")
                
                # Create subdirectory for histograms
                instance_name = os.path.splitext(instance_file)[0]
                histograms_dir = os.path.join(self.results_dir, f"histograms_{instance_name}")
                
                # Save histograms only for GPU backends
                save_hist_this_backend = cfg["save_histograms"] and "nvidia" in self.backend_name.lower()
                if save_hist_this_backend:
                    os.makedirs(histograms_dir, exist_ok=True)
                
                try:
                    # Load instance
                    instance_path = os.path.join(self.istanze_dir, instance_file)
                    instance_data = self.load_instance(instance_path)
                    
                    distance_matrix = instance_data["distance_matrix"]
                    y_true = instance_data["y_true"]
                    info = instance_data["info"]
                    n_points = info["n_samples"]
                    
                    # Find optimal solution
                    optimal_bitstring, optimal_cost = self.find_optimal_solution(
                        distance_matrix, n_points
                    )
                    print(f"  Points: {n_points}, Optimal cost: {optimal_cost:.2f}")
                    
                    # Generate shots grid
                    shots_list = make_shots_grid(n_points, cfg["shots_base"])
                    
                    # Process each seed
                    for seed_idx, seed in enumerate(cfg["seeds"]):
                        theta_cache = {}  # Parameter cache for warm starting
                        
                        for shots in shots_list:
                            coverage = shots / (2 ** n_points)
                            coverage_pct = coverage * 100
                            coverage_str = "< 0.1%" if coverage_pct < 0.1 else f"{coverage_pct:.1f}%"
                            print(f"  Seed {seed}, Shots {shots} (coverage ~{coverage_str})")
                            
                            stop_more_shots = False
                            best_prob_opt_so_far = 0.0
                            no_improve_layers = 0
                            prev_theta = None
                            
                            for layer in cfg["layers"]:
                                try:
                                    # Check if parameters are cached
                                    use_cached = (layer in theta_cache)
                                    msg_tag = " [CACHED θ]" if use_cached else ""
                                    print(f"    Layer {layer}...{msg_tag}", end='', flush=True)
                                    start_time = time.perf_counter()
                                    
                                    # Start system monitor
                                    if cfg["track_metrics"]:
                                        mon_stop, mon_data = start_sys_monitor(cfg["monitor_interval"])
                                    
                                    # Run QAOA
                                    cost, prob_best, ari, theta, results_dict, times, metrics = run_qaoa(
                                        distance_matrix, y_true, n_points, layer, shots, seed,
                                        iterations=cfg["max_iterations"],
                                        theta0=theta_cache.get(layer, prev_theta),
                                        track_metrics=cfg["track_metrics"],
                                        skip_vqe=use_cached
                                    )
                                    
                                    # Update warm start
                                    prev_theta = theta
                                    
                                    # Cache parameters for future shots
                                    if not use_cached:
                                        theta_cache[layer] = theta
                                    
                                    # Stop monitor
                                    if cfg["track_metrics"]:
                                        sys_metrics = stop_sys_monitor(mon_stop, mon_data)
                                    else:
                                        sys_metrics = {}
                                    
                                    exec_time = time.perf_counter() - start_time
                                    
                                    # Calculate additional metrics
                                    add_metrics = calculate_additional_metrics(
                                        results_dict, optimal_cost, shots
                                    )
                                    
                                    # Generate histogram
                                    time_hist = 0.0
                                    if save_hist_this_backend:
                                        time_hist = self.plot_test_histogram(
                                            results_dict, instance_name, n_points,
                                            layer, shots, optimal_cost, histograms_dir,
                                            hist_min_pct=cfg.get("hist_min_pct", 2.0)
                                        )
                                    
                                    # Derived metrics
                                    time_quantum = times['t_quantum']
                                    time_classical = times['t_classic']
                                    
                                    evals_per_sec = (times['n_observe'] / max(times['t_vqe'], 1e-9) 
                                                     if times['t_vqe'] > 0 else 0.0)
                                    shots_per_sec = shots / max(times['t_sampling'], 1e-9)
                                    
                                    # Time-To-Solution calculation
                                    p_succ = max(add_metrics['prob_opt'], 1e-12)
                                    if p_succ >= 1.0:
                                        tts_99 = exec_time
                                    else:
                                        tts_99 = exec_time * np.ceil(np.log(0.01) / np.log(1 - p_succ))
                                    
                                    # Store results
                                    result = {
                                        # Identifiers
                                        'backend': self.backend_name,
                                        'instance': instance_file,
                                        'n_points': n_points,
                                        'layers': layer,
                                        'shots': shots,
                                        'seed': seed,
                                        'seed_idx': seed_idx,
                                        'coverage': coverage,
                                        'coverage_pct': coverage_pct,
                                        
                                        # Solution quality
                                        'optimal_cost': optimal_cost,
                                        'found_cost': cost,
                                        'gap_percent': ((cost - optimal_cost) / optimal_cost * 100 
                                                        if optimal_cost > 0 else 0),
                                        'prob_opt': add_metrics['prob_opt'],
                                        'prob_best': prob_best,
                                        'prob_valid': add_metrics['prob_valid'],
                                        'ari': ari,
                                        
                                        # Timing
                                        'time_total': times['t_total'],
                                        'time_vqe': times['t_vqe'],
                                        'time_observe': times['t_observe'],
                                        'time_sampling': times['t_sampling'],
                                        'time_sample_inner': times['t_sample'],
                                        'exec_time': exec_time,
                                        'time_hist': time_hist,
                                        'time_quantum': time_quantum,
                                        'time_classical': time_classical,
                                        
                                        # Resources
                                        **sys_metrics,
                                        
                                        # Optimizer
                                        'n_energy_evals': times['n_observe'],
                                        'n_sample_calls': times['n_sample'],
                                        'n_observe_calls': times['n_observe'],
                                        'optimizer_iters': times['n_observe'],
                                        'parameters': list(theta),
                                        'theta_norm': np.linalg.norm(theta),
                                        
                                        # Derived metrics
                                        'evals_per_sec': evals_per_sec,
                                        'shots_per_sec': shots_per_sec,
                                        'tts_99': tts_99,
                                        
                                        # Distribution
                                        'n_unique_bitstrings': metrics['n_unique_bitstrings'],
                                        'entropy_samples': add_metrics['entropy_samples'],
                                        'n_opt_states': add_metrics['n_opt_states'],
                                        'max_non_opt_prob': add_metrics['max_non_opt_prob'],
                                        'top_non_opt_bs': add_metrics['top_non_opt_bs'],
                                        
                                        # Metadata
                                        'timestamp_run': datetime.now().isoformat(),
                                        'error': None
                                    }
                                    
                                    all_results.append(result)
                                    
                                    # Stream results
                                    stream_path = append_stream_row(self.backend_name, result)
                                    if stream_path and (layer == cfg["layers"][0]):
                                        print(f"    [STREAM] → {os.path.basename(stream_path)}")
                                    
                                    # Early stopping logic
                                    dom = (add_metrics['prob_opt'] >= cfg['es_P_dom'] and
                                           add_metrics['max_non_opt_prob'] <= cfg['es_Q_others'])
                                    
                                    improved = ((add_metrics['prob_opt'] - best_prob_opt_so_far) >= 
                                                cfg['es_deltaP_min'])
                                    if improved:
                                        best_prob_opt_so_far = add_metrics['prob_opt']
                                        no_improve_layers = 0
                                    else:
                                        no_improve_layers += 1
                                    
                                    if (cfg.get('early_stop', True) and dom and 
                                        no_improve_layers >= cfg['es_patience_layers']):
                                        print(f"    [EARLY-STOP] Stable dominance: "
                                              f"prob_opt={add_metrics['prob_opt']:.3f}, "
                                              f"max_non_opt={add_metrics['max_non_opt_prob']:.3f}")
                                        if cfg.get('es_stop_shots_if_dominant', True):
                                            stop_more_shots = True
                                            print("    [EARLY-STOP] Stopping further shots for this seed.")
                                        break
                                    
                                    print(f" ✓ Cost={cost:.2f}, Prob_opt={add_metrics['prob_opt']:.3f}, "
                                          f"Time_q={time_quantum:.2f}s, Time={exec_time:.1f}s, "
                                          f"Evals/s={evals_per_sec:.1f}")
                                    
                                except Exception as e:
                                    print(f" ✗ Error: {e}")
                                    result = {
                                        'backend': self.backend_name,
                                        'instance': instance_file,
                                        'n_points': n_points,
                                        'layers': layer,
                                        'seed': seed,
                                        'seed_idx': seed_idx,
                                        'shots': shots,
                                        'coverage': coverage,
                                        'error': str(e),
                                        'timestamp_run': datetime.now().isoformat(),
                                    }
                                    all_results.append(result)
                            
                            if stop_more_shots:
                                break
                            
                except Exception as e:
                    print(f"  CRITICAL ERROR: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Create results DataFrame
        df_results = pd.DataFrame(all_results)
        if 'error' not in df_results.columns:
            df_results['error'] = None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save master CSV
        csv_file = os.path.join(self.results_dir, f'qaoa_summary_{timestamp}.csv')
        df_results.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Save Excel with multiple sheets
        excel_file = os.path.join(self.results_dir, f'qaoa_summary_{timestamp}.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Master sheet
            df_results.to_excel(writer, sheet_name='All Results', index=False)
            
            # System info sheet
            df_sys = pd.DataFrame([self.system_info])
            df_sys.to_excel(writer, sheet_name='System Info', index=False)
            
            # Split by backend
            for backend in df_results['backend'].unique():
                df_backend = df_results[df_results['backend'] == backend]
                sheet_name = f'Backend_{backend}'[:31]
                df_backend.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Save separate CSV for backend
                backend_csv = os.path.join(self.results_dir, 
                                           f'qaoa_{backend}_{timestamp}.csv')
                df_backend.to_csv(backend_csv, index=False)
                print(f"Backend {backend} saved to: {backend_csv}")
        
        print(f"Excel saved to: {excel_file}")
        
        # Save per-instance files
        print("\nSaving per-instance files...")
        instance_dir = os.path.join(self.results_dir, f'per_instance_{timestamp}')
        os.makedirs(instance_dir, exist_ok=True)
        
        for instance, df_inst in df_results.groupby('instance'):
            inst_name = os.path.splitext(instance)[0]
            
            # CSV per instance
            inst_csv = os.path.join(instance_dir, f'{inst_name}_{timestamp}.csv')
            df_inst.to_csv(inst_csv, index=False)
            
            # Excel per instance with backend sheets
            inst_excel = os.path.join(instance_dir, f'{inst_name}_{timestamp}.xlsx')
            with pd.ExcelWriter(inst_excel, engine='openpyxl') as writer:
                df_inst.to_excel(writer, sheet_name='All', index=False)
                for backend in df_inst['backend'].unique():
                    df_back = df_inst[df_inst['backend'] == backend]
                    df_back.to_excel(writer, sheet_name=backend[:31], index=False)
        
        print(f"Per-instance files saved to: {instance_dir}")
        
        # Save metadata JSON
        meta_file = os.path.join(self.results_dir, f'meta_run_{timestamp}.json')
        backends_tested = list(df_results['backend'].unique()) if 'backend' in df_results.columns else []
        meta_info = {
            'system_info': self.system_info,
            'config': cfg,
            'timestamp_start': timestamp,
            'timestamp_end': datetime.now().isoformat(),
            'n_instances': len(instances),
            'n_results': len(df_results),
            'backends_tested': backends_tested
        }
        with open(meta_file, 'w') as f:
            json.dump(meta_info, f, indent=2, default=str)
        print(f"Metadata saved to: {meta_file}")
        
        # Best configurations
        best_configs = self.pick_best_configs(df_results)
        if not best_configs.empty:
            best_file = os.path.join(self.results_dir, f'best_configs_{timestamp}.csv')
            best_configs.to_csv(best_file, index=False)
            print(f"Best configurations saved to: {best_file}")
        
        # Generate report
        self.generate_report(df_results)
        
        return df_results
    
    def pick_best_configs(self, df: pd.DataFrame, prob_thr: float = 0.9, 
                          gap_thr: float = 1.0) -> pd.DataFrame:
        """
        Select optimal configuration for each instance based on solution quality and speed.
        
        Args:
            df: Results DataFrame
            prob_thr: Probability threshold for "good" solutions
            gap_thr: Gap percentage threshold for "good" solutions
            
        Returns:
            DataFrame with best configuration per instance
        """
        # Filter valid results (no errors)
        df_valid = df[df['optimal_cost'].notna() & df['error'].isna()].copy()
        if df_valid.empty:
            return pd.DataFrame()
        
        best_list = []
        
        for instance in df_valid['instance'].unique():
            df_inst = df_valid[df_valid['instance'] == instance]
            
            # Find "good" configurations
            good = df_inst[(df_inst['prob_opt'] >= prob_thr) | 
                           (df_inst['gap_percent'] <= gap_thr)]
            
            if not good.empty:
                # Among good solutions, choose fastest
                best = good.sort_values(['time_quantum', 'time_total', 'layers', 'shots']).iloc[0]
            else:
                # Otherwise choose best quality
                best = df_inst.sort_values(['prob_best', 'prob_opt', 'time_quantum'], 
                                            ascending=[False, False, True]).iloc[0]
            
            best_list.append(best)
        
        return pd.DataFrame(best_list)
    
    def generate_report(self, df_results: pd.DataFrame):
        """
        Generate comprehensive summary report of benchmark results.
        
        Args:
            df_results: Complete results DataFrame
        """
        df_valid = df_results[df_results['optimal_cost'].notna() & df_results['error'].isna()].copy()
        
        if len(df_valid) == 0:
            print("No valid results to report")
            return
        
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\nGeneral Statistics:")
        print(f"  Backends tested: {list(df_valid['backend'].unique())}")
        print(f"  Instances processed: {df_valid['instance'].nunique()}")
        print(f"  Total tests: {len(df_valid)}")
        print(f"  Total time: {df_valid['exec_time'].sum():.1f} seconds")
        print(f"  Total quantum time: {df_valid['time_quantum'].sum():.1f} seconds")
        print(f"  Total classical time: {df_valid['time_classical'].sum():.1f} seconds")
        
        # Per-backend summary
        print("\n" + "-"*60)
        print("Performance by backend:")
        backend_summary = df_valid.groupby('backend').agg({
            'prob_opt': 'mean',
            'gap_percent': 'mean',
            'time_quantum': 'mean',
            'time_total': 'mean',
            'exec_time': 'sum'
        }).round(3)
        print(backend_summary)
        
        # Per-instance summary
        summary = df_valid.groupby(['instance', 'n_points']).agg({
            'prob_opt': ['mean', 'max'],
            'prob_best': ['mean', 'max'],
            'prob_valid': ['mean', 'max'],
            'ari': ['mean', 'max'],
            'gap_percent': ['mean', 'min'],
            'time_quantum': ['mean', 'min'],
            'time_classical': ['mean', 'min'],
            'exec_time': 'mean',
            'n_unique_bitstrings': 'mean',
            'entropy_samples': 'mean',
            'evals_per_sec': 'mean',
            'shots_per_sec': 'mean'
        }).round(3)
        
        print("\nSummary by instance:")
        print(summary)
        
        # Performance by shots
        print("\n" + "-"*60)
        print("Performance by shots:")
        shots_summary = df_valid.groupby('shots').agg({
            'prob_opt': 'mean',
            'gap_percent': 'mean',
            'time_quantum': 'mean',
            'coverage_pct': 'mean'
        }).round(3)
        print(shots_summary)
        
        # Best configurations per backend
        print("\n" + "-"*60)
        print("Best configurations by instance and backend:")
        
        for backend in df_valid['backend'].unique():
            print(f"\n--- Backend: {backend} ---")
            df_backend = df_valid[df_valid['backend'] == backend]
            
            for instance in df_backend['instance'].unique():
                subset = df_backend[df_backend['instance'] == instance]
                best_idx = subset['prob_opt'].idxmax()
                best = subset.loc[best_idx]
                print(f"\n{instance}:")
                print(f"  Best: {best['layers']} layers, {best['shots']} shots, seed {best['seed']}")
                print(f"  Prob_opt: {best['prob_opt']:.3f}, Gap: {best['gap_percent']:.2f}%")
                print(f"  Time quantum: {best['time_quantum']:.2f}s, Total: {best['exec_time']:.1f}s")
                tts_str = f"{best.get('tts_99', 0):.1f}s" if best.get('tts_99', float('inf')) != float('inf') else "∞"
                print(f"  Coverage: {best['coverage_pct']:.2f}%, TTS(99%): {tts_str}")
                print(f"  Evals/s: {best['evals_per_sec']:.1f}, Shots/s: {best['shots_per_sec']:.1f}")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main function with command-line argument parsing and optional profiling.
    
    Supports various execution modes including profiling, backend selection,
    and configuration file loading.
    """
    parser = argparse.ArgumentParser(
        description='QAOA Clustering K=2 Benchmark with Comprehensive Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python qaoa_benchmark.py --istanze-dir ./instances
  
  # Run with specific configuration file
  python qaoa_benchmark.py --conf config.json --results-dir ./results
  
  # Test only GPU backend with profiling
  python qaoa_benchmark.py --backend nvidia --profile
  
  # Test multiple backends without metrics
  python qaoa_benchmark.py --backends nvidia qpp-cpu --no-metrics
        """
    )
    
    parser.add_argument('--conf', type=str, 
                        help='Path to JSON configuration file')
    parser.add_argument('--istanze-dir', type=str, 
                        default="./instances",
                        help='Directory containing instance pickle files')
    parser.add_argument('--results-dir', type=str, 
                        default="qaoa_k2_results",
                        help='Directory for output files')
    parser.add_argument('--profile', action='store_true',
                        help='Enable cProfile profiling')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable detailed metrics tracking for faster execution')
    parser.add_argument('--backend', type=str, default=None,
                        help='Test only this backend (overrides configuration)')
    parser.add_argument('--backends', type=str, nargs='+',
                        help='List of backends to test (overrides configuration)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.conf and os.path.exists(args.conf):
        with open(args.conf, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.conf}")
    
    # Override metrics tracking if requested
    if args.no_metrics:
        config['track_metrics'] = False
    
    # Override backends if specified
    if args.backend:
        config['backends'] = [args.backend]
    elif args.backends:
        config['backends'] = args.backends
    
    # Create benchmark instance
    benchmark = QAOAClusteringBenchmark(args.istanze_dir, args.results_dir)
    
    # Run with optional profiling
    if args.profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
        df_results = benchmark.run_benchmark(config)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        
        # Save profile
        profile_file = os.path.join(args.results_dir, 
                                    f'profile_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(profile_file, 'w') as f:
            f.write(s.getvalue())
        print(f"\nProfiling results saved to: {profile_file}")
    else:
        df_results = benchmark.run_benchmark(config)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED!")
    print("="*80)
    
    # Print suggestions for advanced profiling
    print("\nFor advanced profiling:")
    print(f"  1. Flamegraph with py-spy:")
    print(f"     py-spy record -o profile.svg -- python {__file__} --istanze-dir ...")
    print(f"  2. Memory profiling:")
    print(f"     memray run -o profile.bin python {__file__} --istanze-dir ...")
    print(f"  3. Line profiling (requires line_profiler):")
    print(f"     kernprof -l -v {__file__} --istanze-dir ...")

if __name__ == "__main__":
    main()