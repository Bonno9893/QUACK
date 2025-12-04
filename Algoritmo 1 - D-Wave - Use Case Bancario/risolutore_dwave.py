"""
Risolutore D-Wave per il problema di espansione del cluster (Algoritmo 1).

Questo modulo prepara e invia la formulazione QUBO del caso d’uso
bancario al quantum annealer di D-Wave utilizzando l’Ocean SDK.
A partire da un’istanza salvata su file, costruisce il BinaryQuadraticModel,
applica un embedding automatico sulla topologia dell’hardware (inclusa
la scelta di una chain strength tramite uniform_torque_compensation),
imposta i parametri principali di annealing e raccoglie i campioni
restituiti dal QPU.

I risultati vengono salvati in forma tabellare per consentire il
confronto, su ciascuna istanza, tra soluzioni quantistiche e soluzioni
ottenute con i risolutori classici.
"""

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

token="IL TUO TOKEN"

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
    print("Unif_torq:"+str(uniform_torque_compensation(bqm, embedding=None)))
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


      sampler = DWaveCliqueSampler(solver={"topology__type":'zephyr'},token=token)

      if chainstrenght:
        start_time = time.time()
        sampleset = sampler.sample(bqm,
                                label='Example - Clustering - SimulatedAnn',
                                num_reads=n_iteration,
                                return_embedding=True,
                                anneal_schedule = schmol_anneal_schedule,
                                chain_strength= cs_value)
                                #annealing_time = 10)
      else:
        start_time = time.time()
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
from dwave.system import EmbeddingComposite, DWaveSampler
import dwave.inspector
from dimod import BinaryQuadraticModel
import numpy as np
#neal simulated annealing
from neal import SimulatedAnnealingSampler

def cluster_points_composite(n_coordinates, d, indexs_i0, n_points_to_add, l2, quantum,n_iteration,inspector,chainstrenght, cs_value):
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
    bqm.scale(10)
    bqm_values = np.array(list(bqm.to_qubo()[0].values()))
    print("BQM:",bqm_values)
    print("max BQM", max(abs(bqm_values)))
    print("Unif_torq:"+str(uniform_torque_compensation(bqm, embedding=None, prefactor=1.5)))
    #cs_value = max(abs(bqm_values))
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
      sampler = EmbeddingComposite(DWaveSampler(solver={"topology__type":'zephyr'},token=token),)
      #sampler = EmbeddingComposite(DWaveSampler(token=token),)
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

#best_sample,bqm,sampler = cluster_points_v3(d.shape[0],d,indexs_i0,n_points_to_add,10.4,True,50)

import pandas as pd

feasible = None
data_out = {}
outList = []
l2 = [
7.190476,
17.19048,
37.19048,
67.19048,
11,
27.19048,
47.19048,
101.3657,
17.19048,
27.19048,
57.19048,
131,
27.19048,
57.19048,
107.1905,
247.1905,
7.190476,
11,
21,
51,
7.190476,
21,
37.19048,
77.19048,
11,
27.19048,
57.19048,
101,
21,
51,
97.19048,
207.1905,
3.917647,
7.190476,
11,
31,
7.190476,
17.19048,
27.19048,
57.19048,
11,
17.19048,
37.19048,
71,
17.19048,
47.19048,
77.19048,
177.1905,
11,
21,
47.19048,
97.19048,
17.19048,
31,
67.19048,
131,
17.19048,
37.19048,
77.19048,
151,
31,
71,
147.1905,
287.1905,
7.190476,
17.19048,
27.19048,
67.19048,
17.19048,
27.19048,
57.19048,
101,
17.19048,
31,
61,
127.1905,
27.19048,
57.19048,
127.1905,
261,
7.190476,
7.190476,
17.19048,
37.19048,
7.190476,
17.19048,
37.19048,
77.19048,
11,
27.19048,
47.19048,
101,
27.19048,
57.19048,
117.1905,
221,
7.190476,
17.19048,
37.19048,
67.19048,
11,
27.19048,
51,
108.2889,
17.19048,
31,
54.76471,
122.5015,
27.19048,
57.19048,
107.1905,
231,
7.190476,
11,
27.19048,
51,
11,
17.19048,
41,
87.19048,
11,
27.19048,
47.19048,
97.19048,
27.19048,
47.19048,
97.19048,
211,
7.190476,
7.190476,
17.19048,
27.19048,
7.190476,
17.19048,
27.19048,
61,
11,
17.19048,
41,
87.19048,
21,
47.19048,
87.19048,
197.1905,
182.27,
267.29,
312.55,
592.42,
137.19,
207.20,
272.19,
561,
81,
151,
207.19,
457.19,
380.45,
525.61,
671.37,
1162.74,
281.95,
444.90,
538.56,
1044.57,
167.19,
307.19,
417.45,
961,
198.33,
277.19,
341.68,
597.49,
137.19,
217.19,
277.21,
517.48,
77.19,
131,
217.19,
447.45,
157.18,
206.44,
261.43,
471.39,
102.28,
167.52,
211.32,
437.45,
57.19,
127.19,
157.19,
357.19]

ist_list=[0]

CS = 5000

for n_ist in range(168,192):
#for n_ist in ist_list:
    ist = carica_istanza(f'Instances-v4/istanza_{n_ist}.pkl')
    d = ist['dm'].values
    indexs_i0 = ist['idx_i0']
    all_index = list(ist['dm'].index)
    indexs_i0 = [all_index.index(e) for e in indexs_i0 if e in indexs_i0]
    n_points_to_add = ist['n_pints_to_add']
    print(ist)

    data_out['inst'] = f'Instances/istanza_{n_ist}.pkl'
    data_out['l2'] = l2[n_ist]

    for i in range(0,10):
        best_sample,bqm,sampler,cs = cluster_points_clique(d.shape[0],d,indexs_i0,n_points_to_add,l2[n_ist],True,100,False,True,CS)
        res = check_solution(best_sample,n_points_to_add+len(indexs_i0),d)



        data_out['iteration'] = i

        data_out['chain_strenght'] = cs

        if res[0] == True:
            data_out['feasible'] = True
        else:
            data_out['feasible'] = False

        data_out['OF'] = res[1]

        for key, value in sampler.info.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            data_out[f"{key}.{sub_key}.{sub_sub_key}"] = sub_sub_value
                    else:
                        data_out[f"{key}.{sub_key}"] = sub_value
            else:
                data_out[key] = value

        outList.append(data_out.copy())

df = pd.DataFrame(outList)

# Esportazione in Excel
df.to_csv(f"output_bacino400_168to191_100runs_Cliq_Cs{CS}.csv", sep=';', decimal=',')
#df.to_csv("output_0_143_50runs_Lmb-cliq.csv", index=False)

df = pd.DataFrame(outList)

# Esportazione in Excel
df.to_excel("output_8_47.xlsx", index=False)

check_solution(best_sample,n_points_to_add+len(indexs_i0),d)

ist = carica_istanza(f'Instances-v4/istanza_{183}.pkl')
ist

sampler.info

n_ist=99
ist = carica_istanza(f'Instances/istanza_{n_ist}.pkl')
d = ist['dm'].values
indexs_i0 = ist['idx_i0']
all_index = list(ist['dm'].index)
indexs_i0 = [all_index.index(e) for e in indexs_i0 if e in indexs_i0]
n_points_to_add = ist['n_pints_to_add']
print(ist)

#SWEEP or Fixed Trials
import matplotlib.pyplot as plt

l2 = 1
l2_fixed_move=1 #for fixed trials
n_iteration = 30  # Total iterations
feasibility_list = []
chain_list = []
chain_value_list = []
objective_function_list = []
l2_values = []

# Track previous feasibility for gradient calculation
prev_feasibility = 0

for i in range(n_iteration):
    best_sample,bqm,sampler,cs = cluster_points_composite(d.shape[0],d,indexs_i0,n_points_to_add,l2,False,50,False,False,1200)
    feasibility, objective_function = check_solution(best_sample, n_points_to_add + len(indexs_i0), d)
    chain_warning=0
    for key, value in sampler.info.items():
        if key=="warnings":
            if "chain" in str(value).lower():
                 chain_warning=1

    for key, value in sampler.info.items():
        if key == "embedding_context":
            chain_strength_value = value.get('chain_strength')

    chain_list.append(chain_warning)
    chain_value_list.append(chain_strength_value)
    # Convert feasibility to 1 (True) or 0 (False) for plotting

    feasibility_int = int(feasibility)
    feasibility_list.append(feasibility_int)
    objective_function_list.append(objective_function)
    #l2_values.append(l2_fixed_move) #for fixed move
    l2_values.append(l2)
    # Print feasibility and l2
    print(feasibility, l2)

    # Update Lagrange multiplier
    l2 += 5
    #l2_fixed_move+=1
    # Update previous feasibility for next gradient calculation
    prev_feasibility = feasibility_int

# Find the minimum feasible solution
min_feasible_value = min(
    [objective_function_list[i] for i in range(len(feasibility_list)) if feasibility_list[i] == 1],
    default=None
)

# Plotting
plt.figure(figsize=(12, 7))

# Plot all points with a line connecting them
plt.plot(l2_values, objective_function_list, color='#DADADA', linestyle='-', linewidth=0.8)

# Separate data by feasibility and chain status
feasible_chain0 = [(l2_values[i], objective_function_list[i]) for i in range(len(feasibility_list))
                   if feasibility_list[i] == 1 and chain_list[i] == 0]
feasible_chain1 = [(l2_values[i], objective_function_list[i]) for i in range(len(feasibility_list))
                   if feasibility_list[i] == 1 and chain_list[i] == 1]
infeasible_chain0 = [(l2_values[i], objective_function_list[i]) for i in range(len(feasibility_list))
                    if feasibility_list[i] == 0 and chain_list[i] == 0]
infeasible_chain1 = [(l2_values[i], objective_function_list[i]) for i in range(len(feasibility_list))
                    if feasibility_list[i] == 0 and chain_list[i] == 1]

# Plot points based on feasibility and chain status - only add label for the first point of each type
# For feasible points without chain breaks
if feasible_chain0:
    l2_vals, obj_vals = zip(*feasible_chain0)
    plt.scatter(l2_vals, obj_vals, color='blue', marker='o', label='Feasible - no chain break')

    # Add chain strength annotations
    for i, (l2, obj) in enumerate(zip(l2_vals, obj_vals)):
        if (i%2) == 0:
            p=-1.7
        else:
            p=1
        index = l2_values.index(l2)  # Find the original index of this point
        cs_value = round(chain_value_list[index],0)
        plt.annotate(f"{cs_value:.0f}",
                     (l2, obj),
                     textcoords="offset points",
                     xytext=(0, p*7),   # 7 points above the point
                     ha='center',     # Horizontally centered
                     fontsize=8)      # Smaller font size

# For feasible points with chain breaks
if feasible_chain1:
    l2_vals, obj_vals = zip(*feasible_chain1)
    plt.scatter(l2_vals, obj_vals, color='goldenrod', marker='v', label='Feasible - chain break')

    # Add chain strength annotations
    for i, (l2, obj) in enumerate(zip(l2_vals, obj_vals)):
        if (i%2) == 0:
            p=-1.7
        else:
            p=1
        index = l2_values.index(l2)  # Find the original index of this point
        cs_value = round(chain_value_list[index],0)
        plt.annotate(f"{cs_value:.0f}",
                     (l2, obj),
                     textcoords="offset points",
                     xytext=(0, p*7),   # 7 points above the point
                     ha='center',     # Horizontally centered
                     fontsize=8)      # Smaller font size

# For infeasible points without chain breaks
if infeasible_chain0:
    l2_vals, obj_vals = zip(*infeasible_chain0)
    plt.scatter(l2_vals, obj_vals, color='red', marker='o', label='Infeasible - no chain break')

    # Add chain strength annotations
    for i, (l2, obj) in enumerate(zip(l2_vals, obj_vals)):
        if (i%2) == 0:
            p=-1.7
        else:
            p=1
        index = l2_values.index(l2)  # Find the original index of this point
        cs_value = round(chain_value_list[index],0)
        plt.annotate(f"{cs_value:.0f}",
                     (l2, obj),
                     textcoords="offset points",
                     xytext=(0, p*7),   # 7 points above the point
                     ha='center',     # Horizontally centered
                     fontsize=8)      # Smaller font size

# For infeasible points with chain breaks
if infeasible_chain1:
    l2_vals, obj_vals = zip(*infeasible_chain1)
    plt.scatter(l2_vals, obj_vals, color='black', marker='v', label='Infeasible - chain break')

    # Add chain strength annotations
    for i, (l2, obj) in enumerate(zip(l2_vals, obj_vals)):
        if (i%2) == 0:
            p=-1.7
        else:
            p=1
        index = l2_values.index(l2)  # Find the original index of this point
        cs_value = round(chain_value_list[index],0)
        plt.annotate(f"{cs_value:.0f}",
                     (l2, obj),
                     textcoords="offset points",
                     xytext=(0, p*7),   # 7 points above the point
                     ha='center',     # Horizontally centered
                     fontsize=8)      # Smaller font size

# Add an "Optimal" green line at the minimum feasible value
if min_feasible_value is not None:
    plt.axhline(y=min_feasible_value, color='green', linestyle='--', linewidth=1.2, label=f'Optimal: {min_feasible_value:.2f}')

# Add labels, title, and legend
plt.title('Affecting the annealing results by changing $lambda$2')
plt.xlabel('$lambda$2')
#plt.xlabel('trials') #for fixed moves (trials)
plt.ylabel('Objective Function')
plt.legend()

# Save the combined plot
plt.savefig("Sweep_Digital_default_99.png")

# plt.show()
