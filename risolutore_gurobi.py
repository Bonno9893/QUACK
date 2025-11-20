"""
Risolutore Gurobi per il problema di espansione del cluster (Algoritmo 1).

Questo modulo costruisce e risolve con Gurobi il modello intero misto
equivalente alla formulazione QUBO del caso d’uso bancario, partendo
da una singola istanza salvata su file. La soluzione ottenuta (valore
ottimo e log di esecuzione) viene usata come benchmark classico per
confrontare le prestazioni del quantum annealer D-Wave e del risolutore
basato su Simulated Annealing.

I risultati principali (solver, stato, best bound, best solution, gap,
tempo) vengono aggiunti al file results_gurobi.csv, uno per istanza
analizzata.
"""


import gurobipy as gp
import time
import orloge as ol
import csv
import os


def write_results(filePath, logFilePath,time):
    def convert_decimal(value):
        if isinstance(value, float):  # Controlla se è un float
            return str(value).replace('.', ',')  # Sostituisce il punto con una virgola
        return value

    # Leggere il file di log
    log_data = ol.get_info_solver(logFilePath, 'GUROBI')
    data = [filePath,log_data['solver'],log_data['status'],log_data['best_bound'],
            log_data['best_solution'],log_data['gap'],time]
    header = ['fileName','solver','status','best_bound','best_solution','gap','time']
    res_file = "results_gurobi.csv"
    # Controlla se il file esiste
    file_exists = os.path.isfile(res_file)

    # Scrittura nel file CSV
    with open(res_file, mode='a', newline='', encoding='utf-8') as file:  # 'a' per aggiungere i dati
        writer = csv.writer(file,delimiter=';')

        # Scrivi l'intestazione solo se il file non esiste
        if not file_exists:
            writer.writerow(header)

        # Prepara i dati convertendo i float
        converted_data = [convert_decimal(value) for value in data]

        # Scrivi i dati
        writer.writerow(converted_data)


def get_solution(prob):
    all_vars = prob.getVars()
    values = prob.getAttr("X", all_vars)
    names = prob.getAttr("VarName", all_vars)
    solution = []
    for name, val in zip(names, values):
        nameSplit = name.split('_')
        if (nameSplit[0] == 'X') and (val > 0.5):
            solution.append(int(nameSplit[1]))
    return solution

def cluster_points_gpy(timeLimit, logFile, I, I0, dist_matrix, T):

    prob = gp.Model("v4.5")
    prob.Params.LogFile = f"{logFile}"
    prob.Params.TimeLimit = timeLimit
    prob.params.OutputFlag = 1

    # Variables
    X = {}

    for i in range(I):
        X[i] = prob.addVar(vtype=gp.GRB.BINARY, name=f"X_{i}")

    # Objective Function
    prob.setObjective(gp.quicksum(dist_matrix[i][j] * X[i] * X[j] for i in range(I) for j in range(I)), gp.GRB.MINIMIZE)

    # Constraint_1
    prob.addConstr(gp.quicksum(X[i] for i in range(I)) >= T,
                   name=f"Constraint_1")

    # Constraint_2
    for i in I0:
        prob.addConstr(X[i] == 1, name=f"Constraint_2_{i}")

    start_time = time.time()
    prob.optimize()
    end_time = time.time()-start_time
    print(end_time)
    #result = ol.get_info_solver(prob.Params.LogFile, 'GUROBI')
    solution = get_solution(prob)
    print("Objective value:"+str(prob.objVal))
    best_sample = {}
    for i in range(I):
        if i in solution:
            best_sample[i] = 1
        else:
            best_sample[i] = 0

    return best_sample, {}, end_time
