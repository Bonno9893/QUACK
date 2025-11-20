"""
Ottimizzazione adattiva del parametro lambda per il QUBO dell’Algoritmo 1.

Questo modulo implementa l’algoritmo di ottimizzazione del parametro
di penalità lambda usato nel progetto QUACK per il caso d’uso bancario.
Per ogni istanza del problema di espansione del cluster, lambda viene
aggiornato iterativamente sulla base delle soluzioni restituite da un
risolutore QUBO basato su Simulated Annealing, fino a trovare valori
che producono soluzioni fattibili e con buon valore di funzione obiettivo.

Durante l’esecuzione vengono memorizzati i valori di lambda associati
alle soluzioni migliori; il valore finale di lambda per ciascuna istanza
è calcolato come media di questi valori. I lambda utilizzati negli
esperimenti e nel paper sono raccolti nel file lambda.csv e possono
essere riutilizzati direttamente quando si impiegano le istanze .txt
presenti nel repository.
"""

from collections import defaultdict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

def get_groupings(sample):
    """Grab selected items and group them by color"""
    colored_points = defaultdict(list)

    for label, bool_val in sample.items():
        # Skip over items that were not selected
        if not bool_val:
            continue

        # Parse selected items
        # Note: label look like "<x_coord>,<y_coord>_<color>"
        coord, color = label.split("_")
        coord_tuple = tuple(map(float, coord.split(",")))
        colored_points[color].append(coord_tuple)

    return dict(colored_points)


def visualize_groupings(groupings_dict, filename):
    """
    Args:
        groupings_dict: key is a color, value is a list of x-y coordinate tuples.
          For example, {'r': [(0,1), (2,3)], 'b': [(8,3)]}
        filename: name of the file to save plot in
    """
    for color, points in groupings_dict.items():
        # Ignore items that do not contain any coordinates
        if not points:
            continue

        # Populate plot
        point_style = color + "o"
        plt.plot(*zip(*points), point_style)

    plt.savefig(filename)


def visualize_scatterplot(x_y_tuples_list, filename):
    """Plotting out a list of x-y tuples

    Args:
        x_y_tuples_list: A list of x-y coordinate values. e.g. [(1,4), (3, 2)]
    """
    plt.plot(*zip(*x_y_tuples_list), "o")
    plt.savefig(filename)

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # coordinate labels for groups red, green, and blue
        label = "{0},{1}_".format(x, y)
        self.r = label + "r"
        self.b = label + "b"

def get_distance(coordinate_0, coordinate_1):
    diff_x = coordinate_0.x - coordinate_1.x
    diff_y = coordinate_0.y - coordinate_1.y

    return math.sqrt(diff_x**2 + diff_y**2)


def get_max_distance(coordinates):
    max_distance = 0
    for i, coord0 in enumerate(coordinates[:-1]):
        for coord1 in coordinates[i+1:]:
            distance = get_distance(coord0, coord1)
            max_distance = max(max_distance, distance)

    return max_distance

import pickle
def salva_istanza(instance, nome_file):
    with open(nome_file, 'wb') as file:
        pickle.dump(instance, file)
    print(f"Istanza salvata in '{nome_file}'")

def carica_istanza(nome_file):
    # Carica il modello
    with open(nome_file, 'rb') as file:
        models = pickle.load(file)
    print("Istanza caricata con successo.")

    return models

"""## Annealer implementations"""

import math
import random
import dwavebinarycsp
import dwave.inspector
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel

#neal simulated annealing
from neal import SimulatedAnnealingSampler

token="CINE-b8a63847768fc5b69e45edacb4bcc024a10b587f"

def check_embedding(source_edgelist):
    from minorminer import find_embedding
    # Edgelist del grafo BQM
    source_edgelist = [(i, j) for i in idx_i for j in idx_i if i != j]
    # Edgelist del grafo del processore (ad esempio Advantage)
    target_edgelist = DWaveSampler().edgelist

    embedding = find_embedding(source_edgelist, target_edgelist)
    if not embedding:
        print("L'embedding non è possibile per questo problema con il quantum processor selezionato.")
    else:
        print("L'embedding è possibile.")

import time
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system import DWaveCliqueSampler
import dwave.inspector
from dimod import BinaryQuadraticModel
import numpy as np
#neal simulated annealing
from neal import SimulatedAnnealingSampler

def cluster_points_clique(n_coordinates, d, indexs_i0, n_points_to_add, l2, quantum,n_iteration,inspector,chainstrenght,cs_value):
    """Perform clustering analysis on given points

    Args:
        scattered_points (list of tuples):
            Points to be clustered
        filename (str):
            Output file for graphic
        problem_inspector (bool):
            Whether to show problem inspector
    """

    # Tot num of ponts
    n_points = list(range(n_coordinates))

    #indici in I
    idx_i = [i for i in n_points if i not in indexs_i0]

    #indici in I0
    idx_i0 = [i for i in n_points if i in indexs_i0]

    #print(idx_i)
    #print(idx_i0)

    # T
    T = n_points_to_add

    # Initialize BQM (Binary Quadratic Model
    bqm = BinaryQuadraticModel('BINARY')


    # 1. Aggiungi variabili base meno quelle in I0
    for i in idx_i:
      bqm.add_variable(i, 0)

    # Calcolo bi
    b = [0]* len(n_points)
    for i in idx_i:
      for j in idx_i0:
        b[i] += d[i][j]

    # 2. Aggiunge funzione obiettivo (1) + (3.1)
    for i in idx_i:
      for j in idx_i:
        if i != j:
          bqm.add_interaction(i, j, d[i][j]+l2)

    # 3. Aggiunge vincolo (2) + (3.1) linear + (3.2)
    for i in idx_i:
      bqm.add_linear(i,b[i]+l2-(l2*2*T))

    # 6. Aggiunge vincolo (3.1)
    #for i in idx_i:
    #  for j in idx_i:
    #    if i == j:
    #      bqm.add_linear(i,l2)

    # 7. Aggiunge vincolo (3.2)
    #for i in idx_i:
    #  bqm.add_linear(i,-l2*2*T)

    anneal_schedule = [(0.0,0.0),(37.0,0.37),(137.0,0.37),(200.0,1.0)]
    schmol_anneal_schedule = [(0.0,0.0),(3.7,0.37),(13.7,0.37),(20.0,1.0)]
    more_schmol_anneal_schedule = [(0.0,0.0),(0.7,0.37),(2.8,0.37),(4.0,1.0)]
    very_schmol_anneal_schedule = [(0.0,0.0),(0.4,0.37),(1.4,0.37),(2.0,1.0)]
    #bqm.scale(10)
    bqm_values = np.array(list(bqm.to_qubo()[0].values()))
    #print("BQM:",bqm_values)
    #print("max BQM", max(abs(bqm_values)))
    #cs_value = max(abs(bqm_values))
    print("Unif_torq:"+str(uniform_torque_compensation(bqm, embedding=None, prefactor=1.5)))
    #print("Unif_torq:"+str(uniform_torque_compensation(bqm)))
    #cs_value=uniform_torque_compensation(bqm)
    # Submit problem to D-Wave sampler
    #EmbeddingComposite(DWaveSampler(token=token))#SimulatedAnnealingSampler()


    if quantum == False:
      start_time = time.time()
      sampler = SimulatedAnnealingSampler()
      sampleset = sampler.sample(bqm,
                               label='Example - Clustering - SimulatedAnn',
                               num_reads=n_iteration,
                               #anneal_schedule=anneal_schedule)
                               annealing_time = 10)
    else:

      start_time = time.time()
      sampler = DWaveCliqueSampler(solver={"topology__type":'zephyr'},token=token)

      if chainstrenght:
        sampleset = sampler.sample(bqm,
                                label='Example - Clustering - SimulatedAnn',
                                num_reads=n_iteration,
                                return_embedding=True,
                                anneal_schedule = schmol_anneal_schedule,
                                chain_strength= cs_value)
                                #annealing_time = 10)
      else:
        sampleset = sampler.sample(bqm,
                                label='Example - Clustering - SimulatedAnn',
                                num_reads=n_iteration,
                                return_embedding=True,
                                anneal_schedule = schmol_anneal_schedule)


    end_time = time.time()-start_time

    print("ANNEALING TIME:",end_time)
    best_sample = sampleset.first.sample
    for e in idx_i0:
      best_sample[e] = 1
    best_sample = dict(sorted(best_sample.items()))
    #print(best_sample)

    # Extract variables in their order
    variables = list(bqm.variables)
    n = len(variables)

    # Map variables to indices
    var_to_index = {var: idx for idx, var in enumerate(variables)}

    # Initialize the matrix
    qubo_matrix = np.zeros((n, n), dtype=np.float32)

    # Populate the matrix
    qubo, offset = bqm.to_qubo()
    for (u, v), value in qubo.items():
        i, j = var_to_index[u], var_to_index[v]
        qubo_matrix[i, j] = value
        if i != j:
            qubo_matrix[j, i] = value  # Ensure symmetry

    print(qubo_matrix)

    ### get objective function
    # Inizializza una stringa vuota per la funzione obiettivo
    objective_str = ""

    # Aggiungi i termini lineari
    for var, coeff in bqm.linear.items():
        objective_str += f"{coeff}*{var} + "

    # Aggiungi i termini quadratici
    for (var1, var2), coeff in bqm.quadratic.items():
        objective_str += f"{coeff}*{var1}*{var2} + "

    # Rimuovi l'ultimo "+ " dalla stringa
    objective_str = objective_str[:-3]

    # Stampa la funzione obiettivo
    print("Funzione obiettivo:")
    print(objective_str)

    if quantum and inspector:
      dwave.inspector.show(sampleset)

    return best_sample,bqm,sampleset,cs_value

import time
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector

def cluster_points_v3(n_coordinates, d, indexs_i0, n_points_to_add, l2, quantum,n_iteration):
    """Perform clustering analysis on given points

    Args:
        scattered_points (list of tuples):
            Points to be clustered
        filename (str):
            Output file for graphic
        problem_inspector (bool):
            Whether to show problem inspector
    """

    # Tot num of ponts
    n_points = list(range(n_coordinates))

    #indici in I
    idx_i = [i for i in n_points if i not in indexs_i0]

    #indici in I0
    idx_i0 = [i for i in n_points if i in indexs_i0]

    #print(idx_i)
    #print(idx_i0)

    # T
    T = n_points_to_add

    # Initialize BQM (Binary Quadratic Model
    bqm = BinaryQuadraticModel('BINARY')


    # 1. Aggiungi variabili base meno quelle in I0
    for i in idx_i:
      bqm.add_variable(i, 0)

    # Calcolo bi
    b = [0]* len(n_points)
    for i in idx_i:
      for j in idx_i0:
        b[i] += d[i][j]

    # 2. Aggiunge funzione obiettivo (1) + (3.1)
    for i in idx_i:
      for j in idx_i:
        if i != j:
          bqm.add_interaction(i, j, d[i][j]+l2)

    # 3. Aggiunge vincolo (2) + (3.1) linear + (3.2)
    for i in idx_i:
      bqm.add_linear(i,b[i]+l2-(l2*2*T))

    # 6. Aggiunge vincolo (3.1)
    #for i in idx_i:
    #  for j in idx_i:
    #    if i == j:
    #      bqm.add_linear(i,l2)

    # 7. Aggiunge vincolo (3.2)
    #for i in idx_i:
    #  bqm.add_linear(i,-l2*2*T)

    anneal_schedule = [(0.0,0.0),(37.0,0.37),(137.0,0.37),(200.0,1.0)]
    schmol_anneal_schedule = [(0.0,0.0),(3.7,0.37),(13.7,0.37),(20.0,1.0)]
    more_schmol_anneal_schedule = [(0.0,0.0),(0.7,0.37),(2.8,0.37),(4.0,1.0)]
    very_schmol_anneal_schedule = [(0.0,0.0),(0.4,0.37),(1.4,0.37),(2.0,1.0)]
    #print(bqm)
    # Submit problem to D-Wave sampler
    #EmbeddingComposite(DWaveSampler(token=token))#SimulatedAnnealingSampler()


    if quantum == False:
      start_time = time.time()
      sampler = SimulatedAnnealingSampler()
      sampleset = sampler.sample(bqm,
                               label='Example - Clustering - SimulatedAnn',
                               num_reads=n_iteration,
                               seed=random.seed(42),
                               #anneal_schedule=anneal_schedule)
                               annealing_time = 10)
    else:
      start_time = time.time()
      sampler = EmbeddingComposite(DWaveSampler(solver={"topology__type":'zephyr'},token=token),)
      #sampler = EmbeddingComposite(DWaveSampler(token=token),)
      sampleset = sampler.sample(bqm,
                               label='Example - Clustering - SimulatedAnn',
                               num_reads=n_iteration,
                               return_embedding=True,
                               anneal_schedule=schmol_anneal_schedule)
                               #annealing_time = 10)


    end_time = time.time()-start_time

    #print("ANNEALING TIME:",end_time)
    best_sample = sampleset.first.sample
    for e in idx_i0:
      best_sample[e] = 1
    best_sample = dict(sorted(best_sample.items()))
    #print(best_sample)

    # Extract variables in their order
    variables = list(bqm.variables)
    n = len(variables)

    # Map variables to indices
    var_to_index = {var: idx for idx, var in enumerate(variables)}

    # Initialize the matrix
    qubo_matrix = np.zeros((n, n), dtype=np.float32)

    # Populate the matrix
    qubo, offset = bqm.to_qubo()
    for (u, v), value in qubo.items():
        i, j = var_to_index[u], var_to_index[v]
        qubo_matrix[i, j] = value
        if i != j:
            qubo_matrix[j, i] = value  # Ensure symmetry

    #print(qubo_matrix)

    ### get objective function
    # Inizializza una stringa vuota per la funzione obiettivo
    objective_str = ""

    # Aggiungi i termini lineari
    for var, coeff in bqm.linear.items():
        objective_str += f"{coeff}*{var} + "

    # Aggiungi i termini quadratici
    for (var1, var2), coeff in bqm.quadratic.items():
        objective_str += f"{coeff}*{var1}*{var2} + "

    # Rimuovi l'ultimo "+ " dalla stringa
    objective_str = objective_str[:-3]

    # Stampa la funzione obiettivo
    #print("Funzione obiettivo:")
    #print(objective_str)

    #if quantum:
    #  dwave.inspector.show(sampleset)

    return best_sample,bqm

"""## main"""

def calculate_OF(solution,d):
    of = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            of += d[i][j]*solution[i]*solution[j]

    return of

def check_solution(solution, T_plusIo, d):

    of = calculate_OF(solution,d)
    sum_sol = np.sum([int(x) for x in solution.values()])

    if sum_sol == T_plusIo:
        return True, of
    else:
        print(T_plusIo)
        print("punti mancanti",T_plusIo - sum_sol)
        return False, of

#Function definition for digital lambda 2 optimization
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple, Dict

def optimize_l2_parameter(
    d: np.ndarray,
    indexs_i0: list,
    n_points_to_add: int,
    quantum=False,
    n_iteration: int = 1000,
    initial_step: float = 10,
    min_step: float = 0.05,
    decay_rate: float = 0.4,
    patience: int = 4,
    n_runs=50,
    plot_name: str = "l2_optimization_plot.png"
) -> Dict:
    """
    Optimize the l2 parameter for quantum annealing clustering.

    Args:
        d: Distance matrix
        indexs_i0: Initial indices
        n_points_to_add: Number of points to add
        n_iteration: Maximum number of iterations
        initial_step: Initial step size for gradient descent
        min_step: Minimum step size
        decay_rate: Step size decay rate
        patience: Number of iterations without improvement before reducing step size
        plot_name: Name of the output plot file

    Returns:
        Dict containing:
            - best_l2: Optimal l2 value found
            - execution_time: Total execution time in seconds
            - n_iterations: Number of iterations performed
    """
    # Start timing
    start_time = time.time()

    # Initial parameters
    l2 = 1
    feasibility_list = []
    objective_function_list = []
    l2_values = []

    # Optimization parameters
    step_size = initial_step
    no_improvement_count = 0
    best_feasible_objective = float('inf')
    best_l2 = None
    n_l2 = 1
    found_feasible = False

    # Previous values tracking
    prev_feasibility = 0
    prev_objective = None

    for i in range(n_iteration):
        # Run quantum annealing with current l2
        best_sample,bqm,sampler,cs = cluster_points_clique(d.shape[0],d,indexs_i0,n_points_to_add,l2,quantum,50,False,True,2000)
        feasibility, objective_function = check_solution(best_sample, n_points_to_add + len(indexs_i0), d)

        # Store results
        feasibility_int = int(feasibility)
        feasibility_list.append(feasibility_int)
        objective_function_list.append(objective_function)
        l2_values.append(l2)

        if not feasibility:
            l2 += step_size
            found_feasible = False
            no_improvement_count = 0
        else:
            if not found_feasible:
                found_feasible = True
                step_size *= decay_rate

            if objective_function == best_feasible_objective:
                best_l2 += l2
                n_l2 += 1

            if objective_function < best_feasible_objective:
                best_feasible_objective = objective_function
                best_l2 = l2
                n_l2 = 1
                no_improvement_count = 0
                l2 -= step_size
            else:
                no_improvement_count += 1
                if objective_function > best_feasible_objective:
                    l2 -= step_size
                if no_improvement_count >= patience:
                    step_size = max(min_step, step_size * decay_rate)
                    no_improvement_count = 0

                    if step_size <= min_step:
                        break

        l2 = max(0.1, l2)
        prev_feasibility = feasibility_int
        prev_objective = objective_function

     # Calculate execution time
    execution_time = time.time() - start_time

    # Create the visualization plot
    plt.figure(figsize=(12, 7))

    # Plot connecting line
    plt.plot(l2_values, objective_function_list, color='gray', linestyle='-', linewidth=0.8)

    # Separate and plot feasible/infeasible points
    feasible_points = [(l2_values[i], objective_function_list[i])
                      for i in range(len(feasibility_list)) if feasibility_list[i] == 1]
    infeasible_points = [(l2_values[i], objective_function_list[i])
                        for i in range(len(feasibility_list)) if feasibility_list[i] == 0]

    if feasible_points:
        plt.scatter(*zip(*feasible_points), color='blue', label='Feasible (1)', marker='o')
    if infeasible_points:
        plt.scatter(*zip(*infeasible_points), color='red', label='Infeasible (0)', marker='x')

    if best_feasible_objective != float('inf'):
        plt.axhline(y=best_feasible_objective, color='green', linestyle='--', linewidth=1.2,
                   label=f'Optimal: {best_feasible_objective:.2f} (l2={best_l2/n_l2:.4f})')

    plt.title('Affecting the annealing results by changing $lambda$2')
    plt.xlabel('$lambda$2')
    plt.ylabel('Objective Function')
    plt.legend()
    plt.savefig(plot_name)
    plt.close()



    return {
        "best_l2": best_l2/n_l2 if best_l2 is not None else None,
        "execution_time": execution_time,
        "n_iterations": i + 1
    }

# Example usage:
# result = optimize_l2_parameter(
#     d=your_distance_matrix,
#     indexs_i0=your_initial_indices,
#     n_points_to_add=your_n_points,
#     plot_name="custom_plot_name.png"
# )
# print(f"Best l2: {result['best_l2']}")
# print(f"Execution time: {result['execution_time']:.2f} seconds")
# print(f"Number of iterations: {result['n_iterations']}")

"""## EXPERIMENT SIM ANNEALING"""

ist = carica_istanza("Instances/istanza_127.pkl")
d = ist['dm'].values
indexs_i0 = ist['idx_i0']
n_points_to_add = ist['n_pints_to_add']

"""# Start Experiment #

## FOLDER BATCH EXECUTION ##
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, Any

def process_instance(file_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process a single instance file and return the results.

    Args:
        file_path: Path to the .pkl file
        output_dir: Directory where plot files will be saved

    Returns:
        Dictionary containing the results and metadata
    """
    try:
        # Load instance
        ist = carica_istanza(file_path)
        d = ist['dm'].values
        indexs_i0 = ist['idx_i0']
        all_index = list(ist['dm'].index)
        indexs_i0 = [all_index.index(e) for e in indexs_i0 if e in indexs_i0]
        print(indexs_i0)
        n_points_to_add = ist['n_pints_to_add']

        # Generate plot name based on input file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_name = os.path.join(output_dir, f"{base_name}_plot.png")

        # Run optimization
        result = optimize_l2_parameter(
            d=d,
            indexs_i0=indexs_i0,
            n_points_to_add=n_points_to_add,
            quantum=True, #set false for classical optimization
            plot_name=plot_name,
            n_runs=3000 #remove for classical optimization
        )

        # Add metadata
        result['file_name'] = base_name
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['status'] = 'success'

        return result

    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'error',
            'error_message': str(e),
            'best_l2': None,
            'execution_time': None,
            'n_iterations': None
        }

def process_folder(input_folder: str, output_folder: str, log_file: str):
    """
    Process all .pkl files in a folder and log results.

    Args:
        input_folder: Path to folder containing .pkl files
        output_folder: Path to folder for output plots
        log_file: Path to CSV log file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize or load CSV file
    csv_exists = os.path.exists(log_file)
    fieldnames = ['file_name', 'timestamp', 'status', 'best_l2',
                 'execution_time', 'n_iterations', 'error_message']

    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')  # Changed delimiter to tab
        if not csv_exists:
            writer.writeheader()

    # Process each .pkl file
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    total_files = len(pkl_files)

    for idx, file_name in enumerate(pkl_files, 1):
        print(f"\nProcessing file {idx}/{total_files}: {file_name}")
        file_path = os.path.join(input_folder, file_name)

        # Process instance
        result = process_instance(file_path, output_folder)

        # Log results
        with open(log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')  # Changed delimiter to tab
            writer.writerow(result)

        # Print results if successful
        if result['status'] == 'success':
            print(f"Best l2: {result['best_l2']}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            print(f"Number of iterations: {result['n_iterations']}")
        else:
            print(f"Error processing file: {result['error_message']}")

# Example usage
#INPUT_FOLDER = "Instances"
INPUT_FOLDER = "Single_instance_input"
OUTPUT_FOLDER = "Single_instance_output"
LOG_FILE = "processing_results_single.csv"

process_folder(INPUT_FOLDER, OUTPUT_FOLDER, LOG_FILE)
