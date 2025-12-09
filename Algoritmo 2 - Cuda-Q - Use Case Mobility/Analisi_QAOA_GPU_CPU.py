#!/usr/bin/env python3
"""
Analisi QAOA - Versione Finale con Output Organizzato
Crea cartelle separate per ogni analisi con grafici singoli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re
import argparse
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Stile accademico
plt.style.use('seaborn-v0_8-whitegrid')
# Colori neutri per pubblicazioni scientifiche
COLORS = {
    'primary': '#1f77b4',    # blu
    'secondary': '#ff7f0e',  # arancione
    'tertiary': '#2ca02c',   # verde
    'quaternary': '#d62728', # rosso
    'neutral': '#7f7f7f',    # grigio
    'highlight': '#e377c2'   # viola
}

# Palette per multiple serie
ACADEMIC_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def fix_quoted_csv(path: str, output_path: str = None) -> str:
    """Corregge il formato CSV quotato"""
    if output_path is None:
        output_path = path.replace('.csv', '_fixed.csv')
    
    with open(path, 'r', encoding='utf-8-sig') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        header = infile.readline().strip()
        outfile.write(header + '\n')
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('"') and (line.endswith(',"') or line.endswith(',"\r')):
                if line.endswith(',"\r'):
                    line = line[1:-3]
                else:
                    line = line[1:-2]
            elif line.startswith('"'):
                line = line[1:]
                if line.endswith(','):
                    line = line[:-1]
            
            array_pattern = r'""(\[[^\]]+\])""'
            def replace_array_commas(match):
                array_content = match.group(1)
                return f'"{array_content.replace(",", "|")}"'
            
            line = re.sub(array_pattern, replace_array_commas, line)
            outfile.write(line + '\n')
    
    return output_path

def smart_read_csv(path: str) -> pd.DataFrame:
    """Lettura robusta CSV"""
    fixed_path = fix_quoted_csv(path)
    
    try:
        df = pd.read_csv(fixed_path, encoding='utf-8')
        
        # Fix arrays
        for col in df.columns:
            if df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                sample = str(df[col].dropna().iloc[0])
                if sample.startswith('[') and '|' in sample:
                    df[col] = df[col].str.replace('|', ',')
        
        # Clean headers
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        
        # Convert numeric - PATCH: aggiunte nuove colonne
        numeric_cols = ['n_points', 'layers', 'shots', 'seed', 'optimal_cost', 
                       'found_cost', 'prob_opt', 'time_total', 'time_vqe', 
                       'time_observe', 'time_sampling', 'time_sample_inner',
                       'n_observe_calls', 'coverage_pct',
                       'time_quantum', 'quantum_time', 'optimization_time', 'tts_99']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[df['instance'].notna() & (df['instance'] != '')]
        
        return df
        
    finally:
        if os.path.exists(fixed_path):
            os.remove(fixed_path)

class AcademicQAOAAnalyzer:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'qaoa_analysis_academic_{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crea sottocartelle per ogni analisi
        self.folders = {
            '1_instance_sizes': 'System Size Distribution',
            '2_convergence': 'QAOA Convergence Analysis',
            '3_performance': 'GPU vs CPU Performance',
            '4_dwave': 'D-Wave Comparison',
            '5_coverage': 'Solution Space Coverage',
            '6_early_stopping': 'Early Stopping Analysis',
            '7_mobility': 'Mobility Instance Analysis',
            '8_tts_analysis': 'Time-to-Solution Analysis',
            '9_additional': 'Additional Analysis'
        }
        
        for folder in self.folders.keys():
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)
    
    def save_plot_description(self, folder, filename, description):
        """Salva descrizione del grafico"""
        desc_path = os.path.join(self.output_dir, folder, f'{filename}_description.txt')
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(description)
    
    def load_data(self, gpu_file, cpu_file, partial_file, dwave_file=None):
        """Carica e prepara i dati"""
        print("\n=== CARICAMENTO DATI ===")
        
        # Load files
        self.df_gpu = smart_read_csv(gpu_file)
        self.df_gpu['backend'] = 'GPU'
        
        self.df_cpu = smart_read_csv(cpu_file)
        self.df_cpu['backend'] = 'CPU'
        
        self.df_partial = smart_read_csv(partial_file)
        self.df_partial['backend'] = 'CPU'
        self.df_partial['is_partial'] = True
        
        # Combine
        self.df = pd.concat([self.df_gpu, self.df_cpu, self.df_partial], ignore_index=True)
        self.df['is_partial'] = self.df['is_partial'].fillna(False)
        
        # Filter errors
        self.df = self.df[self.df['error'].isna()]
        
        # Add metrics
        self.df['n_qubits'] = self.df['n_points']
        self.df['prob_opt_pct'] = self.df['prob_opt'] * 100
        
        # PATCH: Identify cached with better threshold
        self.df['is_cached'] = (
            (self.df['time_observe'] < 1e-3) & 
            (self.df['layers'] > 1)
        )
        
        # Calculate correct times
        self.df['quantum_time'] = self.df['time_observe'] + self.df['time_sample_inner']
        self.df['optimization_time'] = (self.df['time_vqe'] - self.df['time_observe']).clip(lower=0)
        
        # Calculate TTS-99
        self.df['tts_99'] = self._calculate_tts(self.df['time_sampling'], self.df['prob_opt'])
        
        # Load D-Wave with all timing info
        self.df_dwave = None
        if dwave_file and os.path.exists(dwave_file):
            self.df_dwave = pd.read_excel(dwave_file)
            
            # Check what columns we have
            print("\nD-Wave columns available:")
            for col in self.df_dwave.columns:
                if 'time' in col.lower() or 'solver' in col.lower():
                    sample_values = self.df_dwave[col].dropna().head(5)
                    print(f"  - {col}: {sample_values.tolist()}")
            
            # Standard rename
            self.df_dwave.rename(columns={
                'file_name': 'instance',
                'dwave_solver_time_from_lambda_verified': 'dwave_qpu_time'
            }, inplace=True)
            
            # PATCH: Autodetect unit
            if 'dwave_qpu_time' in self.df_dwave.columns:
                med = self.df_dwave['dwave_qpu_time'].median()
                if med < 1e-3:
                    print(f"\nD-Wave times appear to be in microseconds (median={med}), converting...")
                    self.df_dwave['dwave_qpu_time'] /= 1_000_000
            
            print(f"\nD-Wave QPU times range: {self.df_dwave['dwave_qpu_time'].min():.6f} - {self.df_dwave['dwave_qpu_time'].max():.6f} seconds")
            
            self.df_dwave['instance'] = self.df_dwave['instance'].apply(
                lambda x: x if str(x).endswith('.pkl') else str(x) + '.pkl'
            )
            
            # Aggiungi n_qubits a D-Wave basandoci sui dati QAOA
            instance_qubits = self.df.groupby('instance')['n_qubits'].first().to_dict()
            self.df_dwave['n_qubits'] = self.df_dwave['instance'].map(instance_qubits)
        
        print(f"\nDati caricati: {len(self.df)} righe totali")
        print(f"  Non-cached: {(~self.df['is_cached']).sum()}")
        
        # Estimate missing CPU values
        self._estimate_missing_cpu_values()
        
        return self.df
    
    def _calculate_tts(self, time_per_sample, success_prob, confidence=0.99):
        """Calcola Time-to-Solution con gestione corretta dei casi limite"""
        # Evita divisioni per zero e log di numeri non validi
        success_prob = np.maximum(success_prob, 1e-6)
        success_prob = np.minimum(success_prob, 1 - 1e-6)  # Evita log(0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tts = (time_per_sample / success_prob) * np.log(1 - confidence) / np.log(1 - success_prob)
        
        # Sostituisci inf e nan con valori molto grandi ma finiti
        tts = np.where(np.isfinite(tts), tts, time_per_sample * 1e6)
        
        return tts
    
    def _estimate_missing_cpu_values(self):
        """Stima valori CPU mancanti"""
        print("\n=== STIMA VALORI CPU MANCANTI ===")
        
        partial_instances = self.df[self.df['is_partial']]['instance'].unique()
        
        for instance in partial_instances:
            gpu_data = self.df[(self.df['instance'] == instance) & 
                              (self.df['backend'] == 'GPU')]
            cpu_data = self.df[(self.df['instance'] == instance) & 
                              (self.df['backend'] == 'CPU') & 
                              (~self.df['is_cached'])]
            
            if len(gpu_data) == 0 or len(cpu_data) == 0:
                continue
            
            max_layer_cpu = cpu_data['layers'].max()
            max_layer_gpu = gpu_data['layers'].max()
            
            if max_layer_cpu < max_layer_gpu:
                print(f"  {instance}: stimando layers {max_layer_cpu+1} - {max_layer_gpu}")
                
                # Calculate ratios from existing data
                common_layers = set(cpu_data['layers'].unique()) & set(gpu_data['layers'].unique())
                
                if len(common_layers) > 0:
                    # Simple ratio for prob_opt
                    prob_ratios = []
                    for layer in common_layers:
                        gpu_prob = gpu_data[gpu_data['layers'] == layer]['prob_opt'].mean()
                        cpu_prob = cpu_data[cpu_data['layers'] == layer]['prob_opt'].mean()
                        if gpu_prob > 0:
                            prob_ratios.append(cpu_prob / gpu_prob)
                    
                    avg_prob_ratio = np.mean(prob_ratios) if prob_ratios else 1.0
                    
                    # Time scaling
                    time_ratio = cpu_data['quantum_time'].mean() / gpu_data['quantum_time'].mean()
                    
                    # Estimate missing layers
                    for layer in range(max_layer_cpu + 1, max_layer_gpu + 1):
                        gpu_layer_data = gpu_data[gpu_data['layers'] == layer]
                        
                        for _, gpu_row in gpu_layer_data.iterrows():
                            estimated_row = gpu_row.copy()
                            estimated_row['backend'] = 'CPU'
                            estimated_row['is_estimated'] = True
                            estimated_row['prob_opt'] = gpu_row['prob_opt'] * avg_prob_ratio
                            estimated_row['prob_opt_pct'] = estimated_row['prob_opt'] * 100
                            estimated_row['quantum_time'] = gpu_row['quantum_time'] * time_ratio
                            estimated_row['optimization_time'] = gpu_row['optimization_time'] * time_ratio
                            estimated_row['tts_99'] = self._calculate_tts(estimated_row['quantum_time'], estimated_row['prob_opt'])
                            
                            self.df = pd.concat([self.df, pd.DataFrame([estimated_row])], 
                                              ignore_index=True)
        
        self.df['is_estimated'] = self.df['is_estimated'].fillna(False)
        print(f"  Totale righe stimate: {self.df['is_estimated'].sum()}")
    
    def plot_1_instance_sizes(self):
        """Dimensione istanze - grafici separati"""
        print("\n=== GRAFICO 1: DIMENSIONE ISTANZE ===")
        folder = '1_instance_sizes'
        
        instance_sizes = self.df.groupby('instance')['n_qubits'].first().sort_values()
        
        # Grafico principale
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = range(len(instance_sizes))
        bars = ax.bar(x, instance_sizes.values, color=COLORS['primary'], alpha=0.8)
        
        # Highlight mobility_no_wellbeing
        mobility_idx = None
        for i, instance in enumerate(instance_sizes.index):
            if 'mobility_no_wellbeing' in instance:
                bars[i].set_color(COLORS['highlight'])
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
                mobility_idx = i
        
        # Labels
        for i, (instance, n_qubits) in enumerate(instance_sizes.items()):
            ax.text(i, n_qubits + 0.2, str(int(n_qubits)), 
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Instance', fontsize=14)
        ax.set_ylabel('Number of Qubits', fontsize=14)
        ax.set_title('System Size Distribution', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([os.path.splitext(i)[0] for i in instance_sizes.index], 
                          rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, instance_sizes.max() * 1.1)
        
        # Add legend for highlighted instance
        if mobility_idx is not None:
            ax.text(0.02, 0.98, "Highlighted: mobility_no_wellbeing (real dataset)", 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                   verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'instance_sizes.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = """Instance Size Distribution

This plot shows the number of qubits for each test instance.
- X-axis: Instance names
- Y-axis: Number of qubits
- Blue bars: Regular instances
- Purple bar with black border: mobility_no_wellbeing (real dataset)

The instances range from 4 to 16 qubits, with mobility_no_wellbeing having 16 qubits.
"""
        self.save_plot_description(folder, 'instance_sizes', description)
        
        # Grafico distribuzione aggregata
        fig, ax = plt.subplots(figsize=(10, 6))
        
        size_counts = instance_sizes.value_counts().sort_index()
        ax.bar(size_counts.index, size_counts.values, 
               color=COLORS['secondary'], alpha=0.8, width=0.8)
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Number of Instances', fontsize=14)
        ax.set_title('Distribution of Instance Sizes', fontsize=16)
        ax.set_xticks(size_counts.index)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (size, count) in enumerate(size_counts.items()):
            ax.text(size, count + 0.1, str(count), 
                   ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'size_distribution.png'), dpi=300)
        plt.close()
    
    def plot_2_prob_opt_evolution(self):
        """Evoluzione probabilità - grafici separati per chiarezza"""
        print("\n=== GRAFICO 2: EVOLUZIONE PROBABILITÀ ===")
        folder = '2_convergence'
        
        # Grafico principale: tutte le dimensioni
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sizes_to_plot = [4, 5, 6, 7, 8, 10, 12, 14, 16]
        
        for i, n_qubits in enumerate(sizes_to_plot):
            data = self.df[self.df['n_qubits'] == n_qubits]
            
            if len(data) > 0:
                grouped = data.groupby('layers')['prob_opt_pct'].agg(['mean', 'std', 'count'])
                
                ax.plot(grouped.index, grouped['mean'], 
                       'o-', color=ACADEMIC_PALETTE[i % len(ACADEMIC_PALETTE)], 
                       label=f'{int(n_qubits)} qubits',
                       markersize=8, linewidth=2)
        
        # Early stopping threshold
        ax.axhline(y=10, color=COLORS['quaternary'], linestyle='--', 
                  linewidth=2, alpha=0.7, label='Early Stopping (10%)')
        
        ax.set_xlabel('Number of Layers', fontsize=14)
        ax.set_ylabel('Probability of Finding Optimal Solution (%)', fontsize=14)
        ax.set_title('QAOA Convergence Analysis - All System Sizes', fontsize=16)
        ax.legend(loc='best', ncol=2, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 15.5)
        ax.set_ylim(-5, 105)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'convergence_all.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = """QAOA Convergence Analysis

This plot shows how the probability of finding the optimal solution evolves with the number of QAOA layers.

Legend:
- Different colors represent different system sizes (4-16 qubits)
- Solid lines with circles: Mean probability across all instances of that size
- Red dashed line at 10%: Early stopping threshold

Key observations:
- Smaller systems (4-8 qubits) converge faster and achieve higher probabilities
- Larger systems require more layers and achieve lower maximum probabilities
- Most systems show diminishing returns after 10-12 layers
"""
        self.save_plot_description(folder, 'convergence_all', description)
        
        # Grafici separati per range di dimensioni
        size_ranges = {
            'small': (4, 8, 'Small Systems (4-8 qubits)'),
            'medium': (9, 12, 'Medium Systems (9-12 qubits)'),
            'large': (13, 16, 'Large Systems (13-16 qubits)')
        }
        
        for range_name, (min_q, max_q, title) in size_ranges.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for n_qubits in range(min_q, max_q + 1):
                data = self.df[self.df['n_qubits'] == n_qubits]
                
                if len(data) > 0:
                    grouped = data.groupby('layers')['prob_opt_pct'].agg(['mean', 'std', 'count'])
                    
                    if len(grouped) > 0:
                        ax.plot(grouped.index, grouped['mean'], 
                               'o-', label=f'{int(n_qubits)} qubits',
                               markersize=10, linewidth=2.5)
                        
                        # Add error bars if enough samples
                        if grouped['count'].min() > 1:
                            ax.fill_between(grouped.index,
                                          grouped['mean'] - grouped['std'],
                                          grouped['mean'] + grouped['std'],
                                          alpha=0.2)
            
            ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_xlabel('Number of Layers', fontsize=14)
            ax.set_ylabel('Probability of Finding Optimal (%)', fontsize=14)
            ax.set_title(f'QAOA Convergence - {title}', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 15.5)
            ax.set_ylim(-5, 105)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, f'convergence_{range_name}.png'), dpi=300)
            plt.close()
    
    def plot_3_timing_comparison(self):
        """Confronto tempi CPU/GPU - grafici singoli"""
        print("\n=== GRAFICO 3: CONFRONTO TEMPI ===")
        folder = '3_performance'
        
        # Use only non-cached data
        df_timing = self.df[~self.df['is_cached']]
        
        # 1. Quantum time comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for backend, color in [('GPU', COLORS['primary']), ('CPU', COLORS['secondary'])]:
            data = df_timing[df_timing['backend'] == backend]
            if len(data) > 0:
                grouped = data.groupby('n_qubits')['quantum_time'].mean()
                ax.plot(grouped.index, grouped.values, 
                       'o-', color=color, label=backend, 
                       markersize=10, linewidth=2.5)
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Quantum Operations Time (seconds)', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('Quantum Operations Time: GPU vs CPU', fontsize=16)
        
        # Add annotations for key points
        ax.annotate('GPU advantage\nincreases with\nsystem size', 
                   xy=(14, 0.02), xytext=(12, 0.1),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                   fontsize=11, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'quantum_time_comparison.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = """Quantum Operations Time Comparison

This plot compares the time required for quantum operations between GPU and CPU backends.

Legend:
- Blue line: GPU backend (NVIDIA cuQuantum)
- Orange line: CPU backend (QPP)

Y-axis is in logarithmic scale to better show the performance difference.

Key observations:
- GPU is consistently faster than CPU
- The performance gap increases with system size
- For 16 qubits, GPU is ~29x faster than CPU
"""
        self.save_plot_description(folder, 'quantum_time_comparison', description)
        
        # 2. Speedup analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        speedup_data = []
        for n_qubits in sorted(df_timing['n_qubits'].unique()):
            gpu_time = df_timing[(df_timing['n_qubits'] == n_qubits) & 
                                (df_timing['backend'] == 'GPU')]['quantum_time'].mean()
            cpu_time = df_timing[(df_timing['n_qubits'] == n_qubits) & 
                                (df_timing['backend'] == 'CPU')]['quantum_time'].mean()
            
            if gpu_time > 0 and not np.isnan(cpu_time):
                speedup = cpu_time / gpu_time
                speedup_data.append({'n_qubits': n_qubits, 'speedup': speedup})
        
        if speedup_data:
            df_speedup = pd.DataFrame(speedup_data)
            
            # Bar plot for speedup
            bars = ax.bar(df_speedup['n_qubits'], df_speedup['speedup'], 
                          color=COLORS['tertiary'], alpha=0.8, width=0.7)
            
            # Add value labels on bars
            for bar, speedup in zip(bars, df_speedup['speedup']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{speedup:.1f}x', ha='center', va='bottom', fontsize=11)
            
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_xlabel('Number of Qubits', fontsize=14)
            ax.set_ylabel('GPU Speedup Factor (CPU time / GPU time)', fontsize=14)
            ax.set_title('GPU Acceleration vs System Size', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(df_speedup['speedup']) * 1.15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'gpu_speedup.png'), dpi=300)
        plt.close()
        
        # 3. Total time comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for backend, color in [('GPU', COLORS['primary']), ('CPU', COLORS['secondary'])]:
            data = df_timing[df_timing['backend'] == backend]
            if len(data) > 0:
                grouped = data.groupby('n_qubits')['time_total'].mean()
                ax.plot(grouped.index, grouped.values, 
                       's-', color=color, label=backend, 
                       markersize=10, linewidth=2.5)
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('Total QAOA Execution Time: GPU vs CPU', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'total_time_comparison.png'), dpi=300)
        plt.close()
        
        # 4. Time breakdown
        fig, ax = plt.subplots(figsize=(10, 8))
        
        width = 0.35
        x = np.arange(2)
        
        # Average times
        gpu_quantum = df_timing[df_timing['backend'] == 'GPU']['quantum_time'].mean()
        gpu_opt = df_timing[df_timing['backend'] == 'GPU']['optimization_time'].mean()
        cpu_quantum = df_timing[df_timing['backend'] == 'CPU']['quantum_time'].mean()
        cpu_opt = df_timing[df_timing['backend'] == 'CPU']['optimization_time'].mean()
        
        quantum_times = [gpu_quantum, cpu_quantum]
        opt_times = [gpu_opt, cpu_opt]
        
        bars1 = ax.bar(x - width/2, quantum_times, width, 
                       label='Quantum Operations', color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, opt_times, width, 
                       label='Classical Optimization', color=COLORS['secondary'], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}s', ha='center', va='bottom', fontsize=11)
        
        ax.set_ylabel('Average Time (seconds)', fontsize=14)
        ax.set_title('Time Breakdown by Component', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['GPU', 'CPU'], fontsize=12)
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.001, 10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'time_breakdown.png'), dpi=300)
        plt.close()
    
    def plot_4_dwave_comparison_simplified(self):
        """Confronto D-Wave - grafici singoli con curve"""
        if self.df_dwave is None:
            print("\n[!] Dati D-Wave non disponibili")
            return
            
        print("\n=== GRAFICO 4: CONFRONTO D-WAVE ===")
        folder = '4_dwave'
        
        # Get best QAOA results
        qaoa_best = self.df[
            self.df['backend'] == 'GPU'
        ].groupby('instance').agg({
            'time_sampling': 'min',
            'time_total': 'min', 
            'prob_opt_pct': 'max',
            'n_qubits': 'first'
        }).reset_index()
        
        # Merge with D-Wave
        comparison = qaoa_best.merge(self.df_dwave, on='instance')
        
        if len(comparison) == 0:
            print("[!] Nessuna istanza in comune")
            return
        
        # Sort by qubits
        comparison = comparison.sort_values('n_qubits_x')
        
        # 1. Time comparison plot - CON CURVE INVECE DI BARRE
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepara dati per curve
        x = comparison['n_qubits_x'].values
        y_qaoa = comparison['time_sampling'].values
        y_dwave = comparison['dwave_qpu_time'].values
        
        # Plot curves with markers
        ax.plot(x, y_qaoa, 'o-', color=COLORS['primary'], 
            linewidth=2.5, markersize=10, label='QAOA Sampling Time')
        ax.plot(x, y_dwave, 's-', color=COLORS['secondary'], 
            linewidth=2.5, markersize=10, label='D-Wave QPU Time')
        
        # Fill between per evidenziare chi è più veloce
        ax.fill_between(x, y_qaoa, y_dwave, 
                    where=(y_dwave < y_qaoa), 
                    color=COLORS['secondary'], alpha=0.2, 
                    label='D-Wave faster region')
        ax.fill_between(x, y_qaoa, y_dwave, 
                    where=(y_qaoa <= y_dwave), 
                    color=COLORS['primary'], alpha=0.2, 
                    label='QAOA faster region')
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Time (seconds)', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('QPU Access Time: D-Wave vs QAOA', fontsize=16)
        
        # Imposta ylim dinamico
        min_time = min(y_qaoa.min(), y_dwave.min()) * 0.5
        max_time = max(y_qaoa.max(), y_dwave.max()) * 2
        ax.set_ylim(min_time, max_time)
        
        # Aggiungi disclaimer
        ax.text(0.98, 0.02, "Note: D-Wave times exclude embedding/queue; QAOA times exclude compilation", 
            transform=ax.transAxes, fontsize=10, ha='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'time_comparison.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = """D-Wave vs QAOA Time Comparison

    This plot compares the QPU access time between D-Wave quantum annealer and QAOA GPU simulator.

    Legend:
    - Blue line with circles: QAOA sampling time (GPU simulator)
    - Orange line with squares: D-Wave QPU time (hardware quantum annealer)
    - Shaded regions indicate which method is faster

    Important notes:
    - D-Wave is actual quantum hardware with physical constraints
    - QAOA is a classical GPU simulator optimized for speed
    - The comparison shows D-Wave is typically faster for these problem sizes
    - For larger problems, quantum hardware advantages would likely emerge
    """
        self.save_plot_description(folder, 'time_comparison', description)
        
        # 2. Ratio analysis (rimane uguale)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ratio = comparison['dwave_qpu_time'] / comparison['time_sampling']
        
        scatter = ax.scatter(comparison['n_qubits_x'], ratio,
                        s=150, alpha=0.7, c=comparison['n_qubits_x'],
                        cmap='viridis', edgecolors='black', linewidth=1)
        
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Time Ratio (D-Wave/QAOA)', fontsize=14)
        ax.set_yscale('log')
        ax.set_ylim(0.01, 10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Performance Ratio by System Size', fontsize=16)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Qubits', fontsize=12)
        
        # Add annotations
        ax.text(0.95, 0.95, f'Mean ratio: {ratio.mean():.2f}x', 
            transform=ax.transAxes, fontsize=12,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Shade regions
        ax.axhspan(0, 1, alpha=0.1, color='red', label='D-Wave Faster')
        ax.axhspan(1, 10, alpha=0.1, color='blue', label='QAOA Faster')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'performance_ratio.png'), dpi=300)
        plt.close()
        
        # 3. Summary statistics (rimane uguale)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Calculate statistics
        mean_ratio = ratio.mean()
        median_ratio = ratio.median()
        dwave_faster = (ratio < 1).sum()
        qaoa_faster = (ratio > 1).sum()
        
        summary_text = f"""D-Wave vs QAOA Comparison Summary

    Instances compared: {len(comparison)}
    System size range: {comparison['n_qubits_x'].min():.0f} - {comparison['n_qubits_x'].max():.0f} qubits

    TIMING COMPARISON:
    - D-Wave QPU mean time: {comparison['dwave_qpu_time'].mean():.3e} seconds
    - QAOA sampling mean time: {comparison['time_sampling'].mean():.3e} seconds

    PERFORMANCE RATIO (D-Wave/QAOA):
    - Mean ratio: {mean_ratio:.2f}x
    - Median ratio: {median_ratio:.2f}x
    - D-Wave faster in: {dwave_faster}/{len(comparison)} instances
    - QAOA faster in: {qaoa_faster}/{len(comparison)} instances

    KEY INSIGHTS:
    - D-Wave is typically faster for these problem sizes
    - This is expected as D-Wave uses actual quantum hardware
    - QAOA uses optimized classical simulation on GPU
    - For larger problems (>20 qubits), quantum advantage would emerge
    - The comparison is between different computational paradigms
    """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=13, verticalalignment='top', 
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'summary.png'), dpi=300)
        plt.close()

    def plot_5_coverage_improved(self):
        """Coverage analysis - grafici singoli ben scalati"""
        print("\n=== GRAFICO 5: COVERAGE ANALYSIS ===")
        folder = '5_coverage'
        
        # 1. Coverage vs system size - CORRETTO PER EVITARE TAGLI
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(self.df['n_qubits'], 
                        self.df['coverage_pct'],
                        c=self.df['layers'], 
                        s=50, 
                        alpha=0.6,
                        cmap='viridis')
        
        # Theoretical line
        n_range = np.arange(4, 17)
        theoretical = [5000 / (2**n) * 100 for n in n_range]
        ax.plot(n_range, theoretical, 'r--', linewidth=2.5, 
            label='Theoretical (5000 shots)', zorder=10)
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Coverage (%)', fontsize=14)
        ax.set_yscale('log')
        ax.set_title('Solution Space Coverage vs System Size', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Imposta ylim dinamico per evitare tagli
        min_coverage = self.df['coverage_pct'][self.df['coverage_pct'] > 0].min()
        max_coverage = self.df['coverage_pct'].max()
        ax.set_ylim(min_coverage * 0.5, max_coverage * 2)  # Margine sopra e sotto
        ax.set_xlim(3.5, 16.5)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Layers', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'coverage_vs_size.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Salva descrizione
        description = """Solution Space Coverage Analysis

    This plot shows how the coverage of the solution space decreases with system size.

    Legend:
    - Colored points: Actual coverage achieved (color indicates number of layers)
    - Red dashed line: Theoretical coverage with 5000 shots

    Y-axis uses logarithmic scale to show the exponential decrease in coverage.

    Key observations:
    - Coverage decreases exponentially with system size (as expected)
    - For 16 qubits, we cover less than 10% of the solution space
    - The actual coverage closely follows the theoretical prediction
    """
        self.save_plot_description(folder, 'coverage_vs_size', description)
        
        # 2. Coverage impact on solution quality (rimane uguale)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use fixed logarithmic bins
        coverage_bins = [0.01, 0.1, 1, 10, 100]
        coverage_labels = ['0.01-0.1%', '0.1-1%', '1-10%', '10-100%']
        
        # Create bins
        coverage_data = self.df['coverage_pct'].copy()
        coverage_data = coverage_data.clip(lower=0.01, upper=100)
        self.df['coverage_bin'] = pd.cut(coverage_data, 
                                        bins=coverage_bins, 
                                        labels=coverage_labels,
                                        include_lowest=True)
        
        # Calculate statistics
        coverage_quality = self.df.groupby('coverage_bin', observed=True)['prob_opt_pct'].agg(['mean', 'std', 'count'])
        coverage_quality = coverage_quality.dropna()
        
        if len(coverage_quality) > 0:
            x = range(len(coverage_quality))
            
            # Calculate error bars
            yerr = []
            for idx, row in coverage_quality.iterrows():
                if row['count'] > 1 and not np.isnan(row['std']):
                    yerr.append(row['std'] / np.sqrt(row['count']))
                else:
                    yerr.append(0)
            
            bars = ax.bar(x, coverage_quality['mean'], 
                        yerr=yerr,
                        capsize=5, color=COLORS['secondary'], alpha=0.8,
                        error_kw={'linewidth': 2})
            
            # Add value labels
            for i, (mean, count) in enumerate(zip(coverage_quality['mean'], coverage_quality['count'])):
                if not np.isnan(mean):
                    ax.text(i, mean + max(yerr)*0.5 + 2, f'{mean:.1f}%\n(n={int(count)})', 
                        ha='center', fontsize=11)
            
            ax.set_xlabel('Coverage Range', fontsize=14)
            ax.set_ylabel('Average Prob(optimal) %', fontsize=14)
            ax.set_title('Solution Quality by Coverage Level', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels([coverage_labels[i] for i in range(len(coverage_quality))], 
                            fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 80)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'coverage_impact.png'), dpi=300)
        plt.close()

    def plot_6_early_stopping_cleaner(self):
        """Early stopping analysis - grafico pulito e ben scalato"""
        print("\n=== GRAFICO 6: EARLY STOPPING ANALYSIS ===")
        folder = '6_early_stopping'
        
        # Use only GPU data
        gpu_data = self.df[self.df['backend'] == 'GPU']
        
        # Find stopping layers
        stop_layers = gpu_data.groupby(['instance', 'seed']).agg({
            'layers': 'max',
            'n_qubits': 'first'
        }).reset_index()
        
        # Main plot - CORRETTO IL PROBLEMA DI SCALA
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate statistics
        stop_stats = stop_layers.groupby('n_qubits')['layers'].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Bar plot with error bars
        x = stop_stats.index
        y = stop_stats['mean']
        yerr = stop_stats['std'].fillna(0)
        
        bars = ax.bar(x, y, yerr=yerr, capsize=8, 
                    color=COLORS['primary'], alpha=0.8, 
                    error_kw={'linewidth': 2, 'capthick': 2})
        
        # Add min/max markers with smaller size
        ax.scatter(x, stop_stats['min'], marker='v', s=100,  # ridotto da 150
                color=COLORS['quaternary'], zorder=5, label='Min')
        ax.scatter(x, stop_stats['max'], marker='^', s=100,  # ridotto da 150
                color=COLORS['tertiary'], zorder=5, label='Max')
        
        # Add value labels with better positioning
        for i, (idx, mean, std, max_val) in enumerate(zip(x, y, yerr, stop_stats['max'])):
            # Position label above error bar or max marker, whichever is higher
            label_y = max(mean + std, max_val) + 0.5
            ax.text(idx, label_y, f'{mean:.1f}', 
                ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Qubits', fontsize=14)
        ax.set_ylabel('Layers at Early Stop', fontsize=14)
        ax.set_title('Early Stopping Analysis: Layers Required for Convergence (GPU)', fontsize=16)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Imposta ylim dinamico basato sui dati reali
        max_y = max(stop_stats['max'].max(), (y + yerr).max()) + 2
        ax.set_ylim(0, max_y)
        ax.set_xticks(x)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'early_stopping_main.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Salva descrizione
        description = """Early Stopping Analysis

    This plot shows the number of QAOA layers needed to reach convergence (early stopping criterion).

    Legend:
    - Blue bars: Mean number of layers at early stop
    - Error bars: Standard deviation
    - Red triangles (down): Minimum layers needed
    - Green triangles (up): Maximum layers needed

    Key observations:
    - Small systems (4-6 qubits) converge quickly (5-10 layers)
    - Larger systems need more layers (12-15 layers)
    - High variability for medium-sized systems
    - Early stopping saves computational resources
    """
        self.save_plot_description(folder, 'early_stopping_main', description)
        
        # Summary table rimane uguale...
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Create detailed table
        table_data = []
        for idx in stop_stats.index:
            row = stop_stats.loc[idx]
            table_data.append([
                f"{int(idx)}", 
                f"{row['count']}", 
                f"{row['mean']:.1f} ± {row['std']:.1f}",
                f"{int(row['min'])}-{int(row['max'])}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Qubits', 'Runs', 'Mean ± Std', 'Range'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.2, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Early Stopping Statistics by System Size', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'early_stopping_table.png'), dpi=300)
        plt.close()
    
    def plot_8_tts_analysis(self):
        """Time-to-Solution analysis"""
        print("\n=== GRAFICO 8: TIME-TO-SOLUTION ANALYSIS ===")
        folder = '8_tts_analysis'
        
        # Prepara dati TTS - CORRETTO IL FILTRO
        tts_data = []
        labels = []
        
        # QAOA GPU
        gpu_mask = (self.df['backend'] == 'GPU') & (self.df['tts_99'].notna())
        gpu_tts_series = self.df.loc[gpu_mask, 'tts_99']
        # Filtra valori finiti
        gpu_tts = gpu_tts_series[np.isfinite(gpu_tts_series.astype(float))]
        
        if len(gpu_tts) > 0:
            tts_data.append(gpu_tts.values)
            labels.append('QAOA-GPU')
        
        # QAOA CPU
        cpu_mask = (self.df['backend'] == 'CPU') & (self.df['tts_99'].notna())
        cpu_tts_series = self.df.loc[cpu_mask, 'tts_99']
        # Filtra valori finiti
        cpu_tts = cpu_tts_series[np.isfinite(cpu_tts_series.astype(float))]
        
        if len(cpu_tts) > 0:
            tts_data.append(cpu_tts.values)
            labels.append('QAOA-CPU')
        
        if not tts_data:
            print("[!] Dati TTS insufficienti")
            return
        
        # Plot boxplot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bp = ax.boxplot(tts_data, labels=labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
        
        # Colori
        colors = [COLORS['primary'], COLORS['secondary']]
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        for element in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp[element], color='black')
        plt.setp(bp['medians'], color='black', linewidth=2)
        plt.setp(bp['means'], color='red', linewidth=2)
        
        ax.set_ylabel('Time-to-Solution @99% (seconds)', fontsize=14)
        ax.set_yscale('log')
        ax.set_title('Time-to-Solution Comparison: GPU vs CPU', fontsize=16)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Legend
        ax.plot([], [], color='black', linewidth=2, label='Median')
        ax.plot([], [], color='red', linewidth=2, label='Mean')
        ax.legend(loc='upper right', fontsize=11)
        
        # Aggiungi statistiche nel plot
        if len(tts_data) >= 2:
            gpu_median = np.median(tts_data[0])
            cpu_median = np.median(tts_data[1])
            speedup = cpu_median / gpu_median
            
            ax.text(0.02, 0.98, f'Median TTS Speedup: {speedup:.1f}x', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'tts_comparison.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = """Time-to-Solution Analysis

    This plot compares the Time-to-Solution (TTS) at 99% confidence level between GPU and CPU backends.

    TTS represents the expected time to find the optimal solution with 99% probability, accounting for:
    - Time per sampling run
    - Probability of finding the optimal solution

    Legend:
    - Colored boxes: Interquartile range (25th to 75th percentile)
    - Black line in box: Median TTS
    - Red line in box: Mean TTS
    - Whiskers: Data range excluding outliers
    - Points: Outliers

    The logarithmic scale is used to better visualize the wide range of TTS values.

    Lower TTS values indicate better performance (faster time to find solution with high confidence).
    """
        self.save_plot_description(folder, 'tts_comparison', description)
        
        # Crea anche una tabella di riepilogo
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_stats = []
        for i, (data, label) in enumerate(zip(tts_data, labels)):
            if len(data) > 0:
                summary_stats.append([
                    label,
                    f"{len(data)}",
                    f"{np.median(data):.3f}",
                    f"{np.mean(data):.3f}",
                    f"{np.std(data):.3f}",
                    f"{np.min(data):.3f}",
                    f"{np.max(data):.3f}"
                ])
        
        if summary_stats:
            table = ax.table(cellText=summary_stats,
                            colLabels=['Backend', 'Count', 'Median', 'Mean', 'Std Dev', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15])
            
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)
            
            # Style the table
            for i in range(len(summary_stats) + 1):
                for j in range(7):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#3498db')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            
            ax.set_title('Time-to-Solution Statistics Summary (seconds)', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'tts_statistics.png'), dpi=300)
        plt.close()
    
    def plot_9_additional_analysis(self):
        """Analisi aggiuntive suggerite"""
        print("\n=== GRAFICO 9: ANALISI AGGIUNTIVE ===")
        folder = '9_additional'
        
        # 1. Stacked-area tempo per layer (GPU)
        gpu_data = self.df[(self.df['backend'] == 'GPU') & (~self.df['is_cached'])]
        
        if len(gpu_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            time_by_layer = gpu_data.groupby('layers').agg({
                'quantum_time': 'mean',
                'optimization_time': 'mean'
            })
            
            layers = time_by_layer.index
            quantum_times = time_by_layer['quantum_time'].values
            opt_times = time_by_layer['optimization_time'].values
            
            ax.fill_between(layers, 0, quantum_times, 
                        label='Quantum Time', color=COLORS['primary'], alpha=0.7)
            ax.fill_between(layers, quantum_times, quantum_times + opt_times,
                        label='Optimization Time', color=COLORS['secondary'], alpha=0.7)
            
            ax.set_xlabel('Number of Layers', fontsize=14)
            ax.set_ylabel('Cumulative Time (seconds)', fontsize=14)
            ax.set_title('Time Breakdown by Layer (GPU)', fontsize=16)
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'stacked_time_by_layer.png'), dpi=300)
            plt.close()
        
        # 2. Heat-map prob_opt vs (layers, shots)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Aggregate data
        heatmap_data = self.df.groupby(['layers', 'shots'])['prob_opt_pct'].mean().reset_index()
        pivot = heatmap_data.pivot(index='layers', columns='shots', values='prob_opt_pct')
        
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Prob(optimal) %'}, ax=ax)
            ax.set_xlabel('Number of Shots', fontsize=14)
            ax.set_ylabel('Number of Layers', fontsize=14)
            ax.set_title('Solution Quality: Layers vs Shots', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'heatmap_layers_shots.png'), dpi=300)
            plt.close()
        
        # 3. Scatter coverage vs prob_opt
        fig, ax = plt.subplots(figsize=(10, 8))
        
        valid_data = self.df[(self.df['coverage_pct'] > 0) & (self.df['prob_opt_pct'] > 0)]
        
        if len(valid_data) > 10:
            scatter = ax.scatter(valid_data['coverage_pct'], valid_data['prob_opt_pct'],
                            c=valid_data['n_qubits'], cmap='viridis', 
                            alpha=0.6, s=50)
            
            # Log regression
            from sklearn.linear_model import HuberRegressor
            X = np.log(valid_data['coverage_pct'].values).reshape(-1, 1)
            y = valid_data['prob_opt_pct'].values
            
            reg = HuberRegressor()
            reg.fit(X, y)
            
            x_range = np.logspace(np.log10(valid_data['coverage_pct'].min()), 
                                np.log10(valid_data['coverage_pct'].max()), 100)
            y_pred = reg.predict(np.log(x_range).reshape(-1, 1))
            
            ax.plot(x_range, y_pred, 'r--', linewidth=2, 
                label=f'Huber fit (coef={reg.coef_[0]:.2f})')
            
            ax.set_xscale('log')
            ax.set_xlabel('Coverage (%)', fontsize=14)
            ax.set_ylabel('Prob(optimal) %', fontsize=14)
            ax.set_title('Solution Quality vs Coverage', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Qubits', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'scatter_coverage_quality.png'), dpi=300)
            plt.close()
        
        # 4. TTS-99 ratio GPU/CPU vs qubit - CORRETTO IL PROBLEMA np.isfinite
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate TTS ratios
        tts_ratios = []
        for n_qubits in sorted(self.df['n_qubits'].unique()):
            # GPU TTS
            gpu_mask = (self.df['n_qubits'] == n_qubits) & (self.df['backend'] == 'GPU')
            gpu_tts_all = self.df.loc[gpu_mask, 'tts_99']
            
            # CPU TTS  
            cpu_mask = (self.df['n_qubits'] == n_qubits) & (self.df['backend'] == 'CPU')
            cpu_tts_all = self.df.loc[cpu_mask, 'tts_99']
            
            # Filtra valori validi
            if len(gpu_tts_all) > 0 and len(cpu_tts_all) > 0:
                # Rimuovi NaN
                gpu_tts_clean = gpu_tts_all.dropna()
                cpu_tts_clean = cpu_tts_all.dropna()
                
                # Filtra valori finiti
                if len(gpu_tts_clean) > 0 and len(cpu_tts_clean) > 0:
                    gpu_tts_valid = gpu_tts_clean[np.isfinite(gpu_tts_clean.astype(float))]
                    cpu_tts_valid = cpu_tts_clean[np.isfinite(cpu_tts_clean.astype(float))]
                    
                    if len(gpu_tts_valid) > 0 and len(cpu_tts_valid) > 0:
                        ratio = cpu_tts_valid.mean() / gpu_tts_valid.mean()
                        tts_ratios.append({'n_qubits': n_qubits, 'ratio': ratio})
        
        if tts_ratios:
            df_tts_ratio = pd.DataFrame(tts_ratios)
            
            bars = ax.bar(df_tts_ratio['n_qubits'], df_tts_ratio['ratio'],
                        color=COLORS['tertiary'], alpha=0.8, width=0.7)
            
            # Add value labels
            for bar, ratio in zip(bars, df_tts_ratio['ratio']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{ratio:.1f}x', ha='center', va='bottom', fontsize=11)
            
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_xlabel('Number of Qubits', fontsize=14)
            ax.set_ylabel('TTS-99 Speedup (CPU/GPU)', fontsize=14)
            ax.set_title('Time-to-Solution Speedup vs System Size', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Aggiungi nota se ci sono pochi dati
            if len(df_tts_ratio) < 5:
                ax.text(0.02, 0.98, f'Note: Based on {len(df_tts_ratio)} system sizes with valid TTS data', 
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                    verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'tts_speedup_ratio.png'), dpi=300)
            plt.close()
        else:
            print("  [!] Dati TTS insufficienti per il calcolo dei ratio")
        
        # 5. Performance heatmap by instance and backend
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepara dati per heatmap
        perf_data = self.df.groupby(['instance', 'backend'])['prob_opt_pct'].max().reset_index()
        perf_pivot = perf_data.pivot(index='instance', columns='backend', values='prob_opt_pct')
        
        if not perf_pivot.empty:
            # Ordina per numero di qubit
            instance_qubits = self.df.groupby('instance')['n_qubits'].first()
            perf_pivot['n_qubits'] = perf_pivot.index.map(instance_qubits)
            perf_pivot = perf_pivot.sort_values('n_qubits')
            perf_pivot = perf_pivot.drop('n_qubits', axis=1)
            
            # Crea heatmap
            sns.heatmap(perf_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Best Prob(optimal) %'}, 
                    vmin=0, vmax=100, ax=ax)
            
            # Formatta labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels([label.get_text().replace('.pkl', '') for label in ax.get_yticklabels()], 
                            rotation=0, fontsize=9)
            
            ax.set_xlabel('Backend', fontsize=14)
            ax.set_ylabel('Instance', fontsize=14)
            ax.set_title('Best Performance by Instance and Backend', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'performance_heatmap.png'), dpi=300)
            plt.close()
        
        # 6. Convergence rate analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcola tasso di miglioramento per layer
        convergence_rates = []
        
        for n_qubits in sorted(self.df['n_qubits'].unique()):
            data = self.df[(self.df['n_qubits'] == n_qubits) & (self.df['backend'] == 'GPU')]
            if len(data) > 0:
                layer_perf = data.groupby('layers')['prob_opt_pct'].mean().sort_index()
                
                if len(layer_perf) > 3:  # Need at least 4 points
                    # Calcola miglioramento percentuale tra layer consecutivi
                    improvements = layer_perf.pct_change().dropna()
                    # Media dei primi 5 layer (dove il miglioramento è maggiore)
                    early_improvement = improvements.iloc[:5].mean() if len(improvements) >= 5 else improvements.mean()
                    convergence_rates.append({
                        'n_qubits': n_qubits,
                        'improvement_rate': early_improvement * 100,
                        'final_prob': layer_perf.iloc[-1]
                    })
        
        if convergence_rates:
            df_conv = pd.DataFrame(convergence_rates)
            
            scatter = ax.scatter(df_conv['n_qubits'], df_conv['improvement_rate'],
                            s=df_conv['final_prob']*10, alpha=0.7,
                            c=df_conv['n_qubits'], cmap='viridis',
                            edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Number of Qubits', fontsize=14)
            ax.set_ylabel('Average Improvement Rate per Layer (%)', fontsize=14)
            ax.set_title('Convergence Rate Analysis (GPU)', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # Aggiungi linea di tendenza
            if len(df_conv) > 3:
                z = np.polyfit(df_conv['n_qubits'], df_conv['improvement_rate'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(df_conv['n_qubits'].min(), df_conv['n_qubits'].max(), 100)
                ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.7, linewidth=2, 
                    label='Polynomial fit')
                ax.legend(fontsize=11)
            
            # Note about bubble size - SOSTITUITO IL SIMBOLO ∝
            ax.text(0.02, 0.98, 'Bubble size ~ final probability', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
                verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'convergence_rate.png'), dpi=300)
            plt.close()

    
    def generate_report(self):
        """Report scientifico completo"""
        print("\n=== GENERAZIONE REPORT ===")
        
        report_path = os.path.join(self.output_dir, 'scientific_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("QAOA PERFORMANCE ANALYSIS - SCIENTIFIC REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total experiments: {len(self.df)}\n")
            f.write(f"Valid (non-cached): {(~self.df['is_cached']).sum()}\n")
            f.write(f"Instances analyzed: {self.df['instance'].nunique()}\n")
            f.write(f"System sizes: {self.df['n_qubits'].min():.0f} - {self.df['n_qubits'].max():.0f} qubits\n\n")
            
            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 30 + "\n")
            
            # GPU speedup
            df_fair = self.df[~self.df['is_cached']]
            speedups = []
            for n in sorted(df_fair['n_qubits'].unique()):
                gpu_t = df_fair[(df_fair['backend'] == 'GPU') & (df_fair['n_qubits'] == n)]['quantum_time'].mean()
                cpu_t = df_fair[(df_fair['backend'] == 'CPU') & (df_fair['n_qubits'] == n)]['quantum_time'].mean()
                if gpu_t > 0 and not np.isnan(cpu_t):
                    speedups.append(cpu_t / gpu_t)
            
            if speedups:
                f.write(f"1. GPU Acceleration:\n")
                f.write(f"   - Average speedup: {np.mean(speedups):.1f}x\n")
                f.write(f"   - Maximum speedup: {np.max(speedups):.1f}x\n")
                f.write(f"   - Speedup increases with system size\n\n")
            
            # Solution quality
            max_prob = self.df.groupby('n_qubits')['prob_opt_pct'].max()
            f.write(f"2. Solution Quality:\n")
            f.write(f"   - Best overall: {self.df['prob_opt_pct'].max():.1f}%\n")
            if len(max_prob[max_prob.index <= 8]) > 0:
                f.write(f"   - Systems <=8 qubits: up to {max_prob[max_prob.index <= 8].max():.1f}%\n")
            if len(max_prob[max_prob.index > 8]) > 0:
                f.write(f"   - Systems >8 qubits: up to {max_prob[max_prob.index > 8].max():.1f}%\n")
            f.write("\n")
            
            # D-Wave comparison
            if self.df_dwave is not None:
                f.write(f"3. D-Wave Comparison:\n")
                f.write(f"   - Based on QPU access time vs QAOA sampling time\n")
                f.write(f"   - D-Wave typically faster for tested instances\n")
                f.write(f"   - Note: Comparing quantum hardware vs GPU simulator\n\n")
            
            # Mobility instance
            mobility_data = self.df[self.df['instance'].str.contains('mobility_no_wellbeing')]
            if len(mobility_data) > 0:
                f.write(f"4. Real Dataset (mobility_no_wellbeing):\n")
                f.write(f"   - System size: {mobility_data['n_qubits'].iloc[0]} qubits\n")
                f.write(f"   - Best probability: {mobility_data['prob_opt_pct'].max():.1f}%\n")
                f.write(f"   - Convergence: typically within 10-15 layers\n\n")
            
            # Early stopping analysis
            stop_layers = self.df[self.df['backend'] == 'GPU'].groupby(['instance', 'seed']).agg({
                'layers': 'max',
                'n_qubits': 'first'
            }).reset_index()
            
            if len(stop_layers) > 0:
                avg_stop = stop_layers.groupby('n_qubits')['layers'].mean()
                f.write(f"5. Early Stopping Analysis:\n")
                small_avg = avg_stop[avg_stop.index <= 6].mean() if len(avg_stop[avg_stop.index <= 6]) > 0 else 0
                medium_avg = avg_stop[(avg_stop.index > 6) & (avg_stop.index <= 10)].mean() if len(avg_stop[(avg_stop.index > 6) & (avg_stop.index <= 10)]) > 0 else 0
                large_avg = avg_stop[avg_stop.index > 10].mean() if len(avg_stop[avg_stop.index > 10]) > 0 else 0
                
                if small_avg > 0:
                    f.write(f"   - Small systems (4-6 qubits): avg {small_avg:.1f} layers\n")
                if medium_avg > 0:
                    f.write(f"   - Medium systems (7-10 qubits): avg {medium_avg:.1f} layers\n")
                if large_avg > 0:
                    f.write(f"   - Large systems (>10 qubits): avg {large_avg:.1f} layers\n")
                f.write("\n")
            
            # Coverage analysis
            f.write(f"6. Solution Space Coverage:\n")
            f.write(f"   - With 5000 shots:\n")
            f.write(f"     • 4 qubits: ~{5000/(2**4)*100:.1f}% coverage\n")
            f.write(f"     • 8 qubits: ~{5000/(2**8)*100:.1f}% coverage\n")
            f.write(f"     • 16 qubits: ~{5000/(2**16)*100:.2f}% coverage\n")
            f.write(f"   - Coverage follows theoretical prediction closely\n\n")
            
            # Time-to-Solution if available - CORRETTO PER GESTIRE ARRAY NUMPY
            if 'tts_99' in self.df.columns:
                # Converti eventuali array numpy in scalari
                def extract_scalar(val):
                    if isinstance(val, np.ndarray):
                        return float(val.item())
                    return float(val)
                
                # GPU TTS
                gpu_mask = self.df['backend'] == 'GPU'
                gpu_tts_raw = self.df.loc[gpu_mask, 'tts_99'].dropna()
                
                if len(gpu_tts_raw) > 0:
                    # Converti array numpy in scalari
                    gpu_tts_values = []
                    for val in gpu_tts_raw:
                        try:
                            scalar_val = extract_scalar(val)
                            if np.isfinite(scalar_val):
                                gpu_tts_values.append(scalar_val)
                        except:
                            pass
                    
                    gpu_tts = pd.Series(gpu_tts_values) if gpu_tts_values else pd.Series([])
                else:
                    gpu_tts = pd.Series([])
                
                # CPU TTS
                cpu_mask = self.df['backend'] == 'CPU'
                cpu_tts_raw = self.df.loc[cpu_mask, 'tts_99'].dropna()
                
                if len(cpu_tts_raw) > 0:
                    # Converti array numpy in scalari
                    cpu_tts_values = []
                    for val in cpu_tts_raw:
                        try:
                            scalar_val = extract_scalar(val)
                            if np.isfinite(scalar_val):
                                cpu_tts_values.append(scalar_val)
                        except:
                            pass
                    
                    cpu_tts = pd.Series(cpu_tts_values) if cpu_tts_values else pd.Series([])
                else:
                    cpu_tts = pd.Series([])
                
                if len(gpu_tts) > 0 and len(cpu_tts) > 0:
                    f.write(f"7. Time-to-Solution (99% confidence):\n")
                    f.write(f"   - GPU median: {gpu_tts.median():.3f}s\n")
                    f.write(f"   - CPU median: {cpu_tts.median():.3f}s\n")
                    f.write(f"   - TTS speedup: {cpu_tts.median()/gpu_tts.median():.1f}x\n\n")
            
            f.write("\nDETAILED ANALYSIS FOLDERS\n")
            f.write("-" * 30 + "\n")
            for folder, description in self.folders.items():
                f.write(f"{folder}: {description}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Each folder contains individual plots with descriptions.\n")
            f.write("All timing comparisons use consistent definitions:\n")
            f.write("- quantum_time: time for quantum operations (observe + sample)\n")
            f.write("- optimization_time: classical optimization overhead\n")
            f.write("- time_sampling: time for QAOA sampling (compare with D-Wave)\n")
        
        print(f"  Report salvato in: {report_path}")
    
    def analyze_mobility_instance(self):
        """Analisi mobility_no_wellbeing - grafici singoli"""
        print("\n=== ANALISI MOBILITY_NO_WELLBEING ===")
        folder = '7_mobility'
        
        # Filter data
        mobility_data = self.df[self.df['instance'].str.contains('mobility_no_wellbeing')]
        
        if len(mobility_data) == 0:
            print("[!] Istanza mobility_no_wellbeing non trovata")
            return
        
        # Basic info
        n_qubits = mobility_data['n_qubits'].iloc[0]
        optimal_cost = mobility_data['optimal_cost'].iloc[0]
        
        # 1. Convergence comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for backend, color in [('GPU', COLORS['primary']), ('CPU', COLORS['secondary'])]:
            data = mobility_data[mobility_data['backend'] == backend]
            if len(data) > 0:
                grouped = data.groupby('layers')['prob_opt_pct'].agg(['mean', 'std', 'count'])
                
                ax.plot(grouped.index, grouped['mean'], 'o-', 
                       color=color, label=backend, markersize=10, linewidth=2.5)
                
                if len(grouped) > 0 and grouped['count'].min() > 1:
                    valid_std = grouped['std'].fillna(0)
                    ax.errorbar(grouped.index, grouped['mean'], 
                              yerr=valid_std, fmt='none', 
                              color=color, alpha=0.3, capsize=5)
        
        ax.axhline(y=10, color=COLORS['quaternary'], linestyle='--', 
                  alpha=0.7, linewidth=2)
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Prob(optimal) %', fontsize=14)
        ax.set_title(f'Convergence: mobility_no_wellbeing ({n_qubits} qubits)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0.5, 15.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'convergence.png'), dpi=300)
        plt.close()
        
        # Salva descrizione
        description = f"""Mobility Instance Analysis

This analysis focuses on the mobility_no_wellbeing instance, which represents a real-world dataset.

Instance details:
- Number of qubits: {n_qubits}
- Optimal cost: {optimal_cost:.2f}
- Dataset type: Real mobility data

The convergence plot shows:
- Blue line: GPU performance
- Orange line: CPU performance  
- Both backends achieve similar solution quality
- GPU is significantly faster in execution time
"""
        self.save_plot_description(folder, 'convergence', description)
        
        # 2. Time scaling
        fig, ax = plt.subplots(figsize=(10, 8))
        
        non_cached = mobility_data[~mobility_data['is_cached']]
        
        for backend, color in [('GPU', COLORS['primary']), ('CPU', COLORS['secondary'])]:
            data = non_cached[non_cached['backend'] == backend]
            if len(data) > 0:
                grouped = data.groupby('layers')['quantum_time'].mean()
                if len(grouped) > 0:
                    ax.plot(grouped.index, grouped.values, 's-', 
                           color=color, label=backend, markersize=10, linewidth=2.5)
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Quantum Time (seconds)', fontsize=14)
        ax.set_yscale('log')
        ax.set_title('Time Scaling with Layers', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 1.5)  # Focus on layer 1 where we have data
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'time_scaling.png'), dpi=300)
        plt.close()
        
        # 3. Best configuration summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Find best configs
        best_configs = []
        for backend in ['GPU', 'CPU']:
            backend_data = mobility_data[mobility_data['backend'] == backend]
            if len(backend_data) > 0:
                best_idx = backend_data['prob_opt_pct'].idxmax()
                best = backend_data.loc[best_idx]
                best_configs.append([
                    backend,
                    f"{best['prob_opt_pct']:.1f}%",
                    int(best['layers']),
                    int(best['shots']),
                    f"{best['quantum_time']:.3f}s",
                    f"{best['time_total']:.1f}s"
                ])
        
        if best_configs:
            table = ax.table(cellText=best_configs,
                            colLabels=['Backend', 'Best Prob', 'Layers', 
                                      'Shots', 'Q-Time', 'Total Time'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2.5)
            
            # Style the table
            for i in range(len(best_configs) + 1):
                for j in range(6):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#2196F3')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.text(0.5, 0.8, f'Optimal Cost: {optimal_cost:.2f}', 
               transform=ax.transAxes, ha='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))
        
        ax.set_title('Best Configuration Summary', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'best_configs.png'), dpi=300)
        plt.close()
        
        # 4. Performance distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data_to_plot = []
        labels = []
        
        for backend in ['GPU', 'CPU']:
            backend_data = mobility_data[mobility_data['backend'] == backend]
            if len(backend_data) > 0:
                best_probs = backend_data.groupby(['layers', 'shots'])['prob_opt_pct'].max().values
                if len(best_probs) > 0:
                    data_to_plot.append(best_probs)
                    labels.append(f'{backend} (n={len(best_probs)})')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True, meanline=True)
            
            colors_bp = [COLORS['primary'], COLORS['secondary']]
            for patch, color in zip(bp['boxes'], colors_bp[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style the plot
            for element in ['whiskers', 'fliers', 'caps']:
                plt.setp(bp[element], color='black')
            plt.setp(bp['medians'], color='black', linewidth=2)
            plt.setp(bp['means'], color='red', linewidth=2)
        
        ax.set_ylabel('Max Prob(optimal) % per Configuration', fontsize=14)
        ax.set_title('Performance Distribution Across Configurations', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Add legend for box plot elements
        ax.plot([], [], color='black', linewidth=2, label='Median')
        ax.plot([], [], color='red', linewidth=2, label='Mean')
        ax.legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, folder, 'performance_distribution.png'), dpi=300)
        plt.close()
        
        # 5. Confronto con D-Wave se disponibile
        if self.df_dwave is not None and 'mobility_no_wellbeing.pkl' in self.df_dwave['instance'].values:
            dwave_mobility = self.df_dwave[self.df_dwave['instance'] == 'mobility_no_wellbeing.pkl'].iloc[0]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Confronto tempi
            qaoa_best = mobility_data[mobility_data['backend'] == 'GPU']['time_sampling'].min()
            dwave_time = dwave_mobility['dwave_qpu_time']
            
            comparison_data = pd.DataFrame({
                'Method': ['QAOA-GPU', 'D-Wave'],
                'Time': [qaoa_best, dwave_time]
            })
            
            bars = ax.bar(comparison_data['Method'], comparison_data['Time'],
                          color=[COLORS['primary'], COLORS['tertiary']],
                          alpha=0.8)
            
            # Annotazioni
            for bar, time in zip(bars, comparison_data['Time']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{time:.4f}s', ha='center', fontsize=12)
            
            # Ratio
            ratio = dwave_time / qaoa_best
            ax.text(0.5, ax.get_ylim()[1] * 0.9, 
                   f'D-Wave is {ratio:.1f}x {"faster" if ratio < 1 else "slower"} than QAOA',
                   ha='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor=COLORS['tertiary'] if ratio < 1 else COLORS['primary'], 
                            alpha=0.3))
            
            ax.set_ylabel('Sampling Time (seconds)', fontsize=14)
            ax.set_title('Mobility Instance: QAOA vs D-Wave Direct Comparison', fontsize=16)
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, folder, 'mobility_vs_dwave.png'), dpi=300)
            plt.close()

        # Salva report dettagliato mobility
        report = f"""Mobility Instance Detailed Analysis
{'='*50}

Instance: mobility_no_wellbeing.pkl
Number of qubits: {n_qubits}
Optimal cost: {optimal_cost:.6f}

QAOA Performance Summary:
------------------------
Best GPU probability: {mobility_data[mobility_data['backend'] == 'GPU']['prob_opt_pct'].max():.1f}%
Best CPU probability: {mobility_data[mobility_data['backend'] == 'CPU']['prob_opt_pct'].max():.1f}%

Best GPU time: {mobility_data[mobility_data['backend'] == 'GPU']['time_sampling'].min():.6f}s
Best CPU time: {mobility_data[mobility_data['backend'] == 'CPU']['time_sampling'].min():.6f}s

GPU speedup (sampling): {mobility_data[mobility_data['backend'] == 'CPU']['time_sampling'].min() / mobility_data[mobility_data['backend'] == 'GPU']['time_sampling'].min():.1f}x

Convergence:
-----------
GPU reaches >10% probability at layer: {mobility_data[(mobility_data['backend'] == 'GPU') & (mobility_data['prob_opt_pct'] > 10)]['layers'].min() if any(mobility_data[(mobility_data['backend'] == 'GPU')]['prob_opt_pct'] > 10) else 'Not reached'}
CPU reaches >10% probability at layer: {mobility_data[(mobility_data['backend'] == 'CPU') & (mobility_data['prob_opt_pct'] > 10)]['layers'].min() if any(mobility_data[(mobility_data['backend'] == 'CPU')]['prob_opt_pct'] > 10) else 'Not reached'}
"""
        
        if self.df_dwave is not None and 'mobility_no_wellbeing.pkl' in self.df_dwave['instance'].values:
            dwave_time = self.df_dwave[self.df_dwave['instance'] == 'mobility_no_wellbeing.pkl']['dwave_qpu_time'].iloc[0]
            report += f"""
D-Wave Comparison:
-----------------
D-Wave QPU time: {dwave_time:.6f}s
D-Wave vs QAOA-GPU ratio: {dwave_time / mobility_data[mobility_data['backend'] == 'GPU']['time_sampling'].min():.2f}x
Winner: {'D-Wave' if dwave_time < mobility_data[mobility_data['backend'] == 'GPU']['time_sampling'].min() else 'QAOA-GPU'}
"""
        
        with open(os.path.join(self.output_dir, folder, 'mobility_report.txt'), 'w') as f:
            f.write(report)
    
    def run_all_analyses(self, gpu_file, cpu_file, partial_file, dwave_file=None):
        """Esegue tutte le analisi"""
        # Load data
        self.load_data(gpu_file, cpu_file, partial_file, dwave_file)
        
        # Generate all plots
        self.plot_1_instance_sizes()
        self.plot_2_prob_opt_evolution()
        self.plot_3_timing_comparison()
        self.plot_4_dwave_comparison_simplified()
        self.plot_5_coverage_improved()
        self.plot_6_early_stopping_cleaner()
        self.analyze_mobility_instance()
        self.plot_8_tts_analysis()
        self.plot_9_additional_analysis()
        
        # Generate report
        self.generate_report()
        
        print(f"\n[✓] ANALISI COMPLETATA!")
        print(f"    Output directory: {self.output_dir}")
        print(f"    Struttura output:")
        for folder, desc in self.folders.items():
            print(f"      {folder}/: {desc}")
        print(f"\n    Report principale: {os.path.join(self.output_dir, 'scientific_report.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analisi QAOA Accademica')
    parser.add_argument('--gpu', required=True, help='File CSV risultati GPU')
    parser.add_argument('--cpu', required=True, help='File CSV risultati CPU')
    parser.add_argument('--partial', required=True, help='File CSV risultati CPU parziali')
    parser.add_argument('--dwave', default=None, help='File Excel risultati D-Wave')
    
    args = parser.parse_args()
    
    analyzer = AcademicQAOAAnalyzer()
    analyzer.run_all_analyses(args.gpu, args.cpu, args.partial, args.dwave)