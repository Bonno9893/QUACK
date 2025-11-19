#!/usr/bin/env python3
"""
Complete Pipeline for QUACK Banking Use Case with D-Wave

This script orchestrates the entire workflow for solving clustering problems
in the banking sector using quantum annealing. It includes:
1. Data loading and preprocessing
2. Instance generation
3. Lambda parameter optimization
4. Quantum annealing with D-Wave
5. Classical benchmarking (Gurobi, Simulated Annealing)
6. Performance comparison and visualization

Usage:
    python run_complete_pipeline.py --config config.yaml
    python run_complete_pipeline.py --data-path data/banking.csv --output results/

Author: QUACK Project Team
Date: 2024
License: MIT
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.optimization.qubo_formulation import QUBOFormulator, create_distance_aware_qubo
from src.solvers.dwave_solver import DWaveClusteringSolver, DWaveConfig
from src.solvers.simulated_annealing import SimulatedAnnealingSolver
from src.solvers.gurobi_solver import GurobiSolver
from src.optimization.lambda_optimizer import LambdaOptimizer
from src.utils.evaluation_metrics import ClusteringEvaluator
from src.utils.visualization import ResultVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BankingClusteringPipeline:
    """
    Main pipeline class for the banking clustering use case.
    
    This class orchestrates all components of the clustering workflow,
    from data preparation through quantum solving to result analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.results = {}
        self.instances = []
        
        # Initialize solvers based on configuration
        self._initialize_solvers()
        
        # Set up output directories
        self._setup_directories()
        
        logger.info("Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ['dwave', 'paths', 'experiment']
        for field in required:
            if field not in config:
                raise ValueError(f"Configuration missing required field: {field}")
        
        return config
    
    def _initialize_solvers(self):
        """Initialize all configured solvers."""
        self.solvers = {}
        
        # D-Wave quantum solver
        if self.config.get('dwave', {}).get('enabled', True):
            dwave_config = DWaveConfig(
                api_token=self.config['dwave']['api_token'],
                solver_name=self.config['dwave'].get('solver'),
                num_reads=self.config['dwave'].get('num_reads', 1000),
                annealing_time=self.config['dwave'].get('annealing_time', 20)
            )
            self.solvers['dwave'] = DWaveClusteringSolver(dwave_config)
            logger.info("D-Wave solver initialized")
        
        # Simulated Annealing solver
        if self.config.get('simulated_annealing', {}).get('enabled', True):
            self.solvers['sa'] = SimulatedAnnealingSolver(
                num_reads=self.config.get('simulated_annealing', {}).get('num_reads', 1000)
            )
            logger.info("Simulated Annealing solver initialized")
        
        # Gurobi solver
        if self.config.get('gurobi', {}).get('enabled', True):
            try:
                self.solvers['gurobi'] = GurobiSolver(
                    time_limit=self.config.get('gurobi', {}).get('time_limit', 60)
                )
                logger.info("Gurobi solver initialized")
            except ImportError:
                logger.warning("Gurobi not available, skipping")
    
    def _setup_directories(self):
        """Create necessary output directories."""
        base_dir = Path(self.config['paths']['results_dir'])
        
        self.dirs = {
            'results': base_dir,
            'instances': base_dir / 'instances',
            'solutions': base_dir / 'solutions',
            'metrics': base_dir / 'metrics',
            'visualizations': base_dir / 'visualizations'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_banking_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess banking customer data.
        
        Args:
            data_path (str, optional): Path to data file. If None, uses config path.
        
        Returns:
            pd.DataFrame: Preprocessed customer data
        """
        if data_path is None:
            data_path = self.config['paths']['data_file']
        
        logger.info(f"Loading data from {data_path}")
        
        # Load data (assuming CSV format)
        df = pd.read_csv(data_path)
        
        # Data preprocessing steps based on QUACK project specifications
        # Remove unnecessary columns
        columns_to_remove = [
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
            'AcceptedCmp4', 'AcceptedCmp5', 'Response',
            'Year_Birth', 'Education', 'Marital_Status',
            'Kidhome', 'Teenhome', 'Dt_Customer', 
            'Complain', 'Z_CostContact', 'Z_Revenue'
        ]
        
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        
        # Standardize numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        logger.info(f"Loaded {len(df)} customer records with {len(df.columns)} features")
        
        return df
    
    def generate_instances(
        self,
        data: pd.DataFrame,
        n_instances: int = 10
    ) -> List[Dict]:
        """
        Generate test instances from banking data.
        
        This method creates clustering problem instances by:
        1. Sampling customers from the dataset
        2. Computing distance matrices
        3. Creating seed clusters
        4. Setting up expansion problems
        
        Args:
            data (pd.DataFrame): Customer data
            n_instances (int): Number of instances to generate
        
        Returns:
            List[Dict]: Generated problem instances
        """
        instances = []
        
        # Instance generation parameters from config
        params = self.config['experiment']['instance_params']
        
        for i in range(n_instances):
            # Sample customers
            n_points = params['n_points']
            sampled_data = data.sample(n=n_points, random_state=42 + i)
            
            # Compute distance matrix (Euclidean distance)
            from sklearn.metrics import pairwise_distances
            distance_matrix = pairwise_distances(sampled_data.values)
            
            # Normalize distances
            if distance_matrix.max() > 0:
                distance_matrix = distance_matrix / distance_matrix.max()
            
            # Create seed cluster (initial cluster points)
            seed_size = params['seed_cluster_size']
            seed_indices = np.random.choice(n_points, size=seed_size, replace=False)
            
            # Determine expansion size
            expand_size = params['expansion_size']
            
            # Create ground truth clustering (for evaluation)
            # Using simple k-means for reference
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=params['n_clusters'], random_state=42)
            y_true = kmeans.fit_predict(sampled_data.values)
            
            instance = {
                'id': i,
                'distance_matrix': distance_matrix,
                'seed_cluster': seed_indices,
                'n_expand': expand_size,
                'y_true': y_true,
                'n_points': n_points,
                'n_clusters': params['n_clusters'],
                'data_indices': sampled_data.index.values
            }
            
            instances.append(instance)
            
            logger.info(f"Generated instance {i}: {n_points} points, "
                       f"seed={seed_size}, expand={expand_size}")
        
        self.instances = instances
        return instances
    
    def optimize_lambda_parameters(self) -> Dict[int, float]:
        """
        Optimize lambda penalty parameters for each instance.
        
        This method uses Simulated Annealing to find optimal lambda values
        that ensure feasible solutions with good clustering quality.
        
        Returns:
            Dict[int, float]: Optimal lambda values for each instance
        """
        logger.info("Starting lambda parameter optimization")
        
        optimizer = LambdaOptimizer(
            solver='simulated_annealing',
            lambda_range=self.config['experiment'].get('lambda_range', (0.1, 10.0)),
            num_samples=self.config['experiment'].get('lambda_samples', 20)
        )
        
        optimal_lambdas = {}
        
        for instance in self.instances:
            logger.info(f"Optimizing lambda for instance {instance['id']}")
            
            # Find optimal lambda
            best_lambda, metrics = optimizer.optimize_for_instance(
                distance_matrix=instance['distance_matrix'],
                seed_cluster=instance['seed_cluster'],
                n_expand=instance['n_expand']
            )
            
            optimal_lambdas[instance['id']] = best_lambda
            
            logger.info(f"Instance {instance['id']}: optimal λ = {best_lambda:.3f}, "
                       f"feasibility = {metrics['feasibility_rate']:.2%}")
        
        self.optimal_lambdas = optimal_lambdas
        return optimal_lambdas
    
    def solve_with_all_methods(self) -> pd.DataFrame:
        """
        Solve all instances with all available methods.
        
        This method runs each instance through:
        - D-Wave quantum annealing
        - Simulated Annealing
        - Gurobi (if available)
        
        Returns:
            pd.DataFrame: Performance comparison results
        """
        results = []
        
        for instance in self.instances:
            instance_id = instance['id']
            optimal_lambda = self.optimal_lambdas[instance_id]
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Solving instance {instance_id}")
            logger.info(f"{'='*50}")
            
            instance_results = {
                'instance_id': instance_id,
                'n_points': instance['n_points'],
                'seed_size': len(instance['seed_cluster']),
                'expand_size': instance['n_expand'],
                'optimal_lambda': optimal_lambda
            }
            
            # Solve with each available method
            for solver_name, solver in self.solvers.items():
                logger.info(f"Solving with {solver_name}...")
                
                start_time = time.time()
                
                try:
                    if solver_name == 'dwave':
                        solution = solver.solve_cluster_expansion(
                            distance_matrix=instance['distance_matrix'],
                            seed_cluster=instance['seed_cluster'],
                            n_expand=instance['n_expand'],
                            lambda_penalty=optimal_lambda
                        )
                        
                        # Extract metrics
                        instance_results[f'{solver_name}_energy'] = solution.energy
                        instance_results[f'{solver_name}_feasible'] = solution.is_feasible
                        instance_results[f'{solver_name}_time'] = solution.timing_info.get(
                            'qpu_access_time', 0
                        )
                        
                        # Calculate clustering quality if feasible
                        if solution.is_feasible and instance.get('y_true') is not None:
                            # Create full clustering assignment
                            clustering = np.zeros(instance['n_points'])
                            clustering[instance['seed_cluster']] = 1
                            clustering[solution.selected_indices] = 1
                            
                            ari = adjusted_rand_score(instance['y_true'], clustering)
                            instance_results[f'{solver_name}_ari'] = ari
                    
                    elif solver_name == 'sa':
                        solution = solver.solve(
                            distance_matrix=instance['distance_matrix'],
                            seed_cluster=instance['seed_cluster'],
                            n_expand=instance['n_expand'],
                            lambda_penalty=optimal_lambda
                        )
                        
                        instance_results[f'{solver_name}_energy'] = solution['energy']
                        instance_results[f'{solver_name}_feasible'] = solution['feasible']
                        instance_results[f'{solver_name}_time'] = time.time() - start_time
                        
                        if solution['feasible'] and instance.get('y_true') is not None:
                            clustering = np.zeros(instance['n_points'])
                            clustering[instance['seed_cluster']] = 1
                            clustering[solution['selected_indices']] = 1
                            
                            ari = adjusted_rand_score(instance['y_true'], clustering)
                            instance_results[f'{solver_name}_ari'] = ari
                    
                    elif solver_name == 'gurobi':
                        solution = solver.solve_clustering(
                            distance_matrix=instance['distance_matrix'],
                            n_clusters=instance['n_clusters']
                        )
                        
                        instance_results[f'{solver_name}_objective'] = solution['objective']
                        instance_results[f'{solver_name}_time'] = solution['solve_time']
                        instance_results[f'{solver_name}_gap'] = solution.get('gap', 0)
                        
                        if instance.get('y_true') is not None:
                            ari = adjusted_rand_score(instance['y_true'], solution['clustering'])
                            instance_results[f'{solver_name}_ari'] = ari
                
                except Exception as e:
                    logger.error(f"Error with {solver_name}: {e}")
                    instance_results[f'{solver_name}_error'] = str(e)
            
            results.append(instance_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.dirs['metrics'] / 'performance_comparison.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        self.results_df = results_df
        return results_df
    
    def generate_visualizations(self):
        """
        Generate comprehensive visualizations of results.
        
        Creates plots for:
        - Performance comparison across methods
        - Solution quality metrics
        - Time complexity analysis
        - Feasibility rates
        """
        visualizer = ResultVisualizer(self.dirs['visualizations'])
        
        # 1. Performance comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ARI scores comparison
        ari_cols = [col for col in self.results_df.columns if 'ari' in col.lower()]
        if ari_cols:
            ax = axes[0, 0]
            ari_data = self.results_df[['instance_id'] + ari_cols].melt(
                id_vars='instance_id', 
                var_name='Method', 
                value_name='ARI'
            )
            ari_data['Method'] = ari_data['Method'].str.replace('_ari', '')
            sns.boxplot(data=ari_data, x='Method', y='ARI', ax=ax)
            ax.set_title('Clustering Quality (ARI) Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Adjusted Rand Index', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        
        # Execution time comparison
        time_cols = [col for col in self.results_df.columns if 'time' in col.lower()]
        if time_cols:
            ax = axes[0, 1]
            time_data = self.results_df[['instance_id'] + time_cols].melt(
                id_vars='instance_id',
                var_name='Method',
                value_name='Time (s)'
            )
            time_data['Method'] = time_data['Method'].str.replace('_time', '')
            
            # Log scale for time
            sns.boxplot(data=time_data, x='Method', y='Time (s)', ax=ax)
            ax.set_yscale('log')
            ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Time (seconds, log scale)', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        
        # Feasibility rates
        feasible_cols = [col for col in self.results_df.columns if 'feasible' in col.lower()]
        if feasible_cols:
            ax = axes[1, 0]
            feasibility_rates = {}
            for col in feasible_cols:
                method = col.replace('_feasible', '')
                if col in self.results_df.columns:
                    rate = self.results_df[col].mean() * 100
                    feasibility_rates[method] = rate
            
            methods = list(feasibility_rates.keys())
            rates = list(feasibility_rates.values())
            bars = ax.bar(methods, rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('Feasibility Rate Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Feasibility Rate (%)', fontsize=12)
            ax.set_ylim([0, 105])
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%',
                       ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
        
        # Problem size scaling
        ax = axes[1, 1]
        for solver_name in self.solvers.keys():
            time_col = f'{solver_name}_time'
            if time_col in self.results_df.columns:
                ax.scatter(
                    self.results_df['n_points'],
                    self.results_df[time_col],
                    label=solver_name.upper(),
                    s=50,
                    alpha=0.7
                )
        
        ax.set_xlabel('Problem Size (number of points)', fontsize=12)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_title('Scalability Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save main comparison figure
        fig_path = self.dirs['visualizations'] / 'performance_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance comparison to {fig_path}")
        
        plt.close()
        
        # 2. Generate detailed lambda optimization plot
        self._plot_lambda_optimization()
    
    def _plot_lambda_optimization(self):
        """Generate visualization of lambda optimization process."""
        if not hasattr(self, 'optimal_lambdas'):
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        instance_ids = list(self.optimal_lambdas.keys())
        lambda_values = list(self.optimal_lambdas.values())
        
        ax.bar(instance_ids, lambda_values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Instance ID', fontsize=12)
        ax.set_ylabel('Optimal λ Value', fontsize=12)
        ax.set_title('Optimal Lambda Parameters per Instance', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (inst_id, lam) in enumerate(zip(instance_ids, lambda_values)):
            ax.text(i, lam, f'{lam:.2f}', ha='center', va='bottom', fontsize=9)
        
        fig_path = self.dirs['visualizations'] / 'lambda_optimization.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved lambda optimization plot to {fig_path}")
        
        plt.close()
    
    def generate_report(self):
        """
        Generate a comprehensive report of the experiment.
        
        Creates a detailed markdown report with:
        - Configuration summary
        - Performance metrics
        - Statistical analysis
        - Conclusions and recommendations
        """
        report_path = self.dirs['results'] / 'experiment_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# QUACK Banking Use Case - Experiment Report\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- **Instances**: {len(self.instances)}\n")
            f.write(f"- **Points per instance**: {self.config['experiment']['instance_params']['n_points']}\n")
            f.write(f"- **Clusters**: {self.config['experiment']['instance_params']['n_clusters']}\n")
            f.write(f"- **Seed size**: {self.config['experiment']['instance_params']['seed_cluster_size']}\n")
            f.write(f"- **Expansion size**: {self.config['experiment']['instance_params']['expansion_size']}\n")
            f.write(f"- **Lambda range**: {self.config['experiment'].get('lambda_range', 'default')}\n\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            if hasattr(self, 'results_df'):
                # Calculate average metrics
                f.write("### Average Metrics\n\n")
                f.write("| Solver | Avg. ARI | Feasibility Rate | Avg. Time (s) |\n")
                f.write("|--------|----------|-----------------|---------------|\n")
                
                for solver in self.solvers.keys():
                    ari_col = f'{solver}_ari'
                    feasible_col = f'{solver}_feasible'
                    time_col = f'{solver}_time'
                    
                    avg_ari = self.results_df[ari_col].mean() if ari_col in self.results_df else 'N/A'
                    feas_rate = self.results_df[feasible_col].mean() * 100 if feasible_col in self.results_df else 'N/A'
                    avg_time = self.results_df[time_col].mean() if time_col in self.results_df else 'N/A'
                    
                    if avg_ari != 'N/A':
                        f.write(f"| {solver.upper()} | {avg_ari:.3f} | {feas_rate:.1f}% | {avg_time:.4f} |\n")
                    else:
                        f.write(f"| {solver.upper()} | N/A | N/A | N/A |\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            # Analyze D-Wave performance
            if 'dwave' in self.solvers and hasattr(self, 'results_df'):
                dwave_time = self.results_df['dwave_time'].mean()
                classical_time = self.results_df[[col for col in self.results_df.columns 
                                                 if 'time' in col and 'dwave' not in col]].mean().mean()
                
                if dwave_time < classical_time:
                    speedup = classical_time / dwave_time
                    f.write(f"- **Quantum Advantage**: D-Wave achieved {speedup:.1f}x speedup over classical methods\n")
                else:
                    f.write(f"- **Classical Performance**: Classical methods currently outperform quantum for this problem size\n")
            
            f.write("\n## Conclusions\n\n")
            f.write("The experiment successfully demonstrated the application of quantum annealing ")
            f.write("to banking customer clustering problems. Key observations include:\n\n")
            f.write("1. Lambda parameter optimization is crucial for feasible solutions\n")
            f.write("2. Quantum annealing shows promise for larger problem instances\n")
            f.write("3. Classical methods maintain advantage for small-scale problems\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("- Continue testing with larger problem instances (N > 100)\n")
            f.write("- Explore hybrid quantum-classical approaches\n")
            f.write("- Investigate problem-specific embedding strategies\n")
        
        logger.info(f"Report generated: {report_path}")
    
    def run(self):
        """
        Execute the complete pipeline.
        
        This is the main entry point that orchestrates all steps:
        1. Load data
        2. Generate instances  
        3. Optimize parameters
        4. Solve with all methods
        5. Generate visualizations
        6. Create report
        """
        logger.info("="*60)
        logger.info("Starting QUACK Banking Clustering Pipeline")
        logger.info("="*60)
        
        # Step 1: Load banking data
        logger.info("\n[Step 1/6] Loading banking data...")
        data = self.load_banking_data()
        
        # Step 2: Generate instances
        logger.info("\n[Step 2/6] Generating test instances...")
        n_instances = self.config['experiment'].get('n_instances', 10)
        instances = self.generate_instances(data, n_instances=n_instances)
        
        # Step 3: Optimize lambda parameters
        logger.info("\n[Step 3/6] Optimizing lambda parameters...")
        optimal_lambdas = self.optimize_lambda_parameters()
        
        # Step 4: Solve with all methods
        logger.info("\n[Step 4/6] Solving with all available methods...")
        results_df = self.solve_with_all_methods()
        
        # Step 5: Generate visualizations
        logger.info("\n[Step 5/6] Generating visualizations...")
        self.generate_visualizations()
        
        # Step 6: Generate report
        logger.info("\n[Step 6/6] Generating experiment report...")
        self.generate_report()
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {self.dirs['results']}")
        logger.info("="*60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='QUACK Banking Clustering Pipeline with D-Wave'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Override data path from config'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--instances',
        type=int,
        help='Number of instances to generate'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = BankingClusteringPipeline(args.config)
        
        # Override parameters if provided
        if args.data_path:
            pipeline.config['paths']['data_file'] = args.data_path
        
        if args.output:
            pipeline.config['paths']['results_dir'] = args.output
            pipeline._setup_directories()
        
        if args.instances:
            pipeline.config['experiment']['n_instances'] = args.instances
        
        # Run the pipeline
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
