#!/usr/bin/env python3
"""
Demo QUACK – Caso d'uso bancario (Algoritmo 1: Espansione del cluster).

Questa demo illustra la pipeline completa per il problema di espansione del cluster
nel contesto del progetto QUACK. Partendo da un'istanza reale (file JSON), costruisce
la formulazione QUBO e la risolve con tre approcci: Simulated Annealing (SA),
Quantum Annealing su D-Wave (QA) e ottimizzazione esatta con Gurobi (MIP).
Al termine viene stampato un riepilogo comparativo delle soluzioni trovate.

Uso:
    python demo_quack_banking.py --instance-id 96 --num-reads 1000
    python demo_quack_banking.py --instance-id 96 --num-reads 500 --verbose
    python demo_quack_banking.py --instance-id 96 --skip-dwave --skip-gurobi
"""

# =============================================================================
# IMPORT
# =============================================================================
import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import dimod

# Simulated Annealing: prova prima 'neal', poi 'dwave.samplers'
try:
    from neal import SimulatedAnnealingSampler
except ImportError:
    from dwave.samplers import SimulatedAnnealingSampler

# D-Wave: disponibile solo se configurato correttamente
try:
    from dwave.system import EmbeddingComposite, DWaveSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

# Gurobi: disponibile solo se installato con licenza valida
try:
    import gurobipy as gp
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# =============================================================================
# FUNZIONI DI SUPPORTO PER LA STAMPA
# =============================================================================

def print_instance_summary(instance: dict, labels: list, verbose: bool) -> None:
    """
    Stampa un riepilogo dell'istanza caricata.

    Parametri:
        instance: dizionario con distance_matrix, seed_indices, candidate_indices, T, n_points
        labels: lista delle etichette originali dei punti (es. ["229", "253", ...])
        verbose: se True, stampa informazioni aggiuntive (statistiche distanze, primi label)
    """
    n = instance["n_points"]
    n_seed = len(instance["seed_indices"])
    n_cand = len(instance["candidate_indices"])
    T = instance["T"]

    print(f"  Punti totali: {n}")
    print(f"  Seed (I0): {n_seed} punto/i")
    print(f"  Candidati: {n_cand} punto/i")
    print(f"  T (punti da aggiungere): {T}")

    if verbose:
        # Mostra i primi 5 label reali (o tutti se meno di 5)
        n_show = min(5, len(labels))
        print(f"  Primi {n_show} label: {labels[:n_show]}")

        # Statistiche sulla matrice delle distanze
        D = instance["distance_matrix"]
        # Estrai solo la parte triangolare superiore (escludendo la diagonale)
        upper_tri = D[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            print(f"  Distanze – min: {upper_tri.min():.4f}, max: {upper_tri.max():.4f}, media: {upper_tri.mean():.4f}")


def print_solver_header(solver_name: str, verbose: bool) -> None:
    """
    Stampa un'intestazione prima dell'esecuzione di un solver (solo in modalità verbose).

    Parametri:
        solver_name: nome del solver (es. "Simulated Annealing (SA)")
        verbose: se True, stampa l'intestazione
    """
    if verbose:
        print(f"\n=== {solver_name} ===")


def print_gap_analysis(results: list, verbose: bool) -> None:
    """
    Calcola e stampa il gap percentuale tra i solver e Gurobi (se disponibile).

    Il gap è definito come: gap(%) = (E_solver - E_Gurobi) / |E_Gurobi| * 100

    NOTA IMPORTANTE: l'energia di SA/QA proviene dal BQM (include la penalità λ),
    mentre l'energia di Gurobi è il puro costo di clustering (somma distanze).
    Il confronto diretto ha senso solo quando la soluzione è fattibile.

    Parametri:
        results: lista di dizionari con i risultati dei solver
        verbose: se True, stampa l'analisi dei gap
    """
    if not verbose:
        return

    # Cerca il risultato di Gurobi
    gurobi_result = next((r for r in results if r["solver"] == "Gurobi"), None)
    if gurobi_result is None or not gurobi_result["feasible"]:
        print("\n[INFO] Gap non calcolabile: Gurobi non disponibile o soluzione non ottima.")
        return

    e_gurobi = gurobi_result["energy"]
    if abs(e_gurobi) < 1e-9:
        print("\n[INFO] Gap non calcolabile: energia Gurobi prossima a zero.")
        return

    print("\n--- Analisi gap rispetto a Gurobi ---")
    print("(Nota: SA/QA usano energia BQM con penalità; Gurobi usa solo distanze)")

    for r in results:
        if r["solver"] == "Gurobi":
            continue
        if not r["feasible"]:
            print(f"  Gap {r['solver']} vs Gurobi: N/A (soluzione non fattibile)")
            continue

        e_solver = r["energy"]
        gap = (e_solver - e_gurobi) / abs(e_gurobi) * 100
        print(f"  Gap {r['solver']} vs Gurobi: {gap:+.2f}%")


# =============================================================================
# CARICAMENTO ISTANZA
# =============================================================================

def load_instance_json(path: Path) -> tuple:
    """
    Carica un'istanza dal file JSON e la converte nel formato interno della demo.

    Formato atteso del file JSON:
        {
            "distance_matrix": [
                {"label1": dist_0_1, "label2": dist_0_2, ...},  // riga 0
                {"label1": dist_1_1, "label2": dist_1_2, ...},  // riga 1
                ...
            ],
            "idx_i0": [label_seed1, label_seed2, ...],  // etichette dei punti seed
            "points_to_add": T,                          // numero di punti da aggiungere
            "selected_coordinates": [...]                // coordinate (non usate nella demo)
        }

    Parametri:
        path: percorso del file JSON

    Restituisce:
        tuple (instance_dict, labels) dove:
            - instance_dict contiene: distance_matrix (numpy), seed_indices, candidate_indices, T, n_points
            - labels è la lista delle etichette originali dei punti

    Note:
        - Le etichette (label) sono stringhe che identificano i clienti nel dataset originale.
        - Gli indici interni (seed_indices, candidate_indices) sono 0-based e si riferiscono
          alla posizione nell'ordine stabilito da labels.
    """
    with open(path, "r") as f:
        data = json.load(f)

    dm_list = data["distance_matrix"]

    # L'ordine dei label è dato dalle chiavi del primo dizionario.
    # Questo ordine definisce il mapping label -> indice 0-based.
    labels = list(dm_list[0].keys())
    n = len(labels)

    # Costruzione della matrice numpy n×n.
    # D[i, j] = distanza tra il punto con indice i e il punto con indice j.
    D = np.zeros((n, n))
    for i, row_dict in enumerate(dm_list):
        for j, lbl in enumerate(labels):
            D[i, j] = row_dict[lbl]

    # Conversione delle etichette seed in indici 0-based.
    # idx_i0 contiene le etichette originali (interi), che vanno cercate in labels.
    seed_labels = data["idx_i0"]
    seed_indices = [labels.index(str(lbl)) for lbl in seed_labels]

    # I candidati sono tutti i punti non appartenenti al seed I0.
    candidate_indices = [i for i in range(n) if i not in seed_indices]

    # T = numero di punti da aggiungere al cluster (vincolo di cardinalità).
    T = data["points_to_add"]

    instance = {
        "distance_matrix": D,
        "seed_indices": seed_indices,
        "candidate_indices": candidate_indices,
        "T": T,
        "n_points": n
    }

    return instance, labels


# =============================================================================
# LETTURA PARAMETRO LAMBDA
# =============================================================================

def read_lambda_for_instance(lambda_csv_path: Path, instance_id: int) -> tuple:
    """
    Legge il file lambda.csv e restituisce il valore di λ e l'ID_Paper per l'istanza richiesta.

    Formato atteso di lambda.csv (formato storico del progetto):
        - Separatore di campo: ';'
        - Separatore decimale: ','
        - Colonne richieste: ID_Paper, ID, LAMBDA

    Convenzione di mapping:
        - instance_id (passato da CLI) corrisponde alla colonna ID
        - ID_Paper viene usato per costruire il nome del file: instance_<ID_Paper>.txt
        - LAMBDA è il valore del parametro di penalità ottimizzato per questa istanza

    Parametri:
        lambda_csv_path: percorso del file lambda.csv
        instance_id: ID dell'istanza (colonna ID nel CSV)

    Restituisce:
        tuple (lambda_value, id_paper) dove:
            - lambda_value (float): valore di λ per la formulazione QUBO
            - id_paper (int): identificativo per il nome del file istanza

    Solleva:
        ValueError: se le colonne richieste non esistono o l'ID non è trovato
    """
    df = pd.read_csv(lambda_csv_path, sep=";", decimal=",")
    df.columns = [c.upper().strip() for c in df.columns]

    # Verifica presenza delle colonne necessarie
    for col in ("ID_PAPER", "ID", "LAMBDA"):
        if col not in df.columns:
            raise ValueError(f"Colonna '{col}' non trovata in {lambda_csv_path}")

    # Cerca la riga corrispondente all'instance_id
    row = df[df["ID"] == instance_id]
    if row.empty:
        raise ValueError(f"Instance ID {instance_id} non trovato in {lambda_csv_path}")

    lambda_val = float(row.iloc[0]["LAMBDA"])
    id_paper = int(row.iloc[0]["ID_PAPER"])

    return lambda_val, id_paper


# =============================================================================
# COSTRUZIONE DEL MODELLO QUBO (BQM)
# =============================================================================

def build_bqm(instance: dict, lambda_value: float) -> dimod.BinaryQuadraticModel:
    """
    Costruisce il BinaryQuadraticModel (BQM) per il problema di espansione del cluster.

    Formulazione QUBO (storica del progetto QUACK):

        min  Σ_{i,j ∈ C, i<j} (d_ij + λ) x_i x_j  +  Σ_{i ∈ C} [b_i + λ - 2λT] x_i

    dove:
        - C = insieme dei candidati (punti non nel seed I0)
        - d_ij = distanza tra i punti i e j
        - λ = parametro di penalità per il vincolo di cardinalità
        - b_i = Σ_{j ∈ I0} d_ij (somma delle distanze dal punto i verso tutti i punti del seed)
        - T = numero di punti da aggiungere
        - x_i ∈ {0, 1} = variabile binaria (1 se il punto i viene selezionato)

    Questa formulazione incorpora:
        1. Il termine di distanza intra-cluster (minimizzare le distanze tra punti selezionati)
        2. Il termine di distanza verso il seed (minimizzare le distanze verso I0)
        3. La penalità quadratica per il vincolo (Σx_i - T)² espansa e semplificata

    Parametri:
        instance: dizionario con distance_matrix, seed_indices, candidate_indices, T
        lambda_value: valore del parametro di penalità λ

    Restituisce:
        dimod.BinaryQuadraticModel in forma BINARY
    """
    d = instance["distance_matrix"]
    seed_idx = instance["seed_indices"]
    cand_idx = instance["candidate_indices"]
    T = instance["T"]

    bqm = dimod.BinaryQuadraticModel("BINARY")

    # Calcolo di b_i: somma delle distanze dal candidato i verso tutti i punti del seed.
    # Questo termine cattura l'affinità di ciascun candidato con il cluster esistente.
    b = {i: sum(d[i][j] for j in seed_idx) for i in cand_idx}

    # Termini quadratici: interazioni tra coppie di candidati.
    # Il coefficiente (d_ij + λ) combina:
    #   - d_ij: costo di distanza se entrambi i punti sono selezionati
    #   - λ: contributo dalla penalità (Σx_i)² = Σ_i x_i² + 2Σ_{i<j} x_i x_j
    for i in cand_idx:
        for j in cand_idx:
            if i < j:
                bqm.add_interaction(i, j, d[i][j] + lambda_value)

    # Termini lineari: coefficiente per ogni variabile x_i.
    # Il coefficiente [b_i + λ - 2λT] combina:
    #   - b_i: distanza cumulata verso il seed
    #   - λ: dalla penalità (Σx_i)² (termine x_i²)
    #   - -2λT: dalla penalità -2T(Σx_i)
    for i in cand_idx:
        bqm.add_linear(i, b[i] + lambda_value - 2 * lambda_value * T)

    return bqm


# =============================================================================
# SOLVER: SIMULATED ANNEALING
# =============================================================================

def solve_sa(bqm: dimod.BinaryQuadraticModel, T: int, num_reads: int = 1000,
             verbose: bool = False) -> dict:
    """
    Risolve il BQM utilizzando Simulated Annealing (SA).

    SA è un algoritmo metaeuristico classico che esplora lo spazio delle soluzioni
    accettando mosse peggiorative con probabilità decrescente (temperatura).
    Viene usato come baseline affidabile per confrontare con QA e Gurobi.

    Parametri:
        bqm: il BinaryQuadraticModel da risolvere
        T: numero atteso di punti da selezionare (per verifica fattibilità)
        num_reads: numero di campionamenti indipendenti
        verbose: se True, stampa informazioni aggiuntive durante l'esecuzione

    Restituisce:
        dict con chiavi: solver, energy, feasible, time, selected

    Note:
        - energy è l'energia del BQM (include la penalità λ)
        - feasible è True se il numero di variabili a 1 è esattamente T
        - time è il tempo CPU totale
    """
    if verbose:
        print(f"  Avvio SA con {num_reads} campionamenti...")

    sampler = SimulatedAnnealingSampler()
    start = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=42)
    elapsed = time.perf_counter() - start

    best = sampleset.first
    # Conta quante variabili sono state impostate a 1 (punti selezionati)
    selected = sum(best.sample.values())
    # La soluzione è fattibile se seleziona esattamente T punti
    feasible = (selected == T)

    if verbose:
        print(f"  Completato in {elapsed:.4f}s – Energia: {best.energy:.4f}, Selezionati: {selected}")

    return {
        "solver": "SA",
        "energy": best.energy,
        "feasible": feasible,
        "time": elapsed,
        "selected": int(selected)
    }


# =============================================================================
# SOLVER: D-WAVE QUANTUM ANNEALER
# =============================================================================

def solve_dwave(bqm: dimod.BinaryQuadraticModel, T: int, num_reads: int = 1000,
                verbose: bool = False) -> dict:
    """
    Risolve il BQM utilizzando il quantum annealer di D-Wave.

    L'annealing quantistico sfrutta effetti di tunneling quantistico per esplorare
    lo spazio delle soluzioni. Richiede un embedding del problema sulla topologia
    fisica del processore (gestito automaticamente da EmbeddingComposite).

    Parametri:
        bqm: il BinaryQuadraticModel da risolvere
        T: numero atteso di punti da selezionare (per verifica fattibilità)
        num_reads: numero di campionamenti sul QPU
        verbose: se True, stampa informazioni aggiuntive durante l'esecuzione

    Restituisce:
        dict con chiavi: solver, energy, feasible, time, selected

    Solleva:
        ImportError: se dwave.system non è disponibile

    Note:
        - energy è l'energia del BQM (include la penalità λ)
        - time è il tempo di accesso al QPU (non include overhead di rete)
    """
    if not DWAVE_AVAILABLE:
        raise ImportError("dwave.system non disponibile")

    if verbose:
        print(f"  Connessione a D-Wave e invio del problema ({num_reads} reads)...")

    sampler = EmbeddingComposite(DWaveSampler())
    start = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads, label="QUACK Demo")
    elapsed = time.perf_counter() - start

    best = sampleset.first
    selected = sum(best.sample.values())
    feasible = (selected == T)

    # Estrai il tempo effettivo di QPU se disponibile nei metadati
    timing = sampleset.info.get("timing", {})
    qpu_time = timing.get("qpu_access_time", elapsed * 1e6) / 1e6  # da µs a s

    if verbose:
        print(f"  Completato – QPU time: {qpu_time:.4f}s, Energia: {best.energy:.4f}, Selezionati: {selected}")

    return {
        "solver": "QA",
        "energy": best.energy,
        "feasible": feasible,
        "time": qpu_time,
        "selected": int(selected)
    }


# =============================================================================
# SOLVER: GUROBI (MIP ESATTO)
# =============================================================================

def solve_gurobi(instance: dict, verbose: bool = False) -> dict:
    """
    Risolve il problema di espansione del cluster con Gurobi (ottimizzazione esatta).

    A differenza di SA e QA, Gurobi risolve un modello MIP equivalente dove:
        - Il vincolo di cardinalità è imposto esattamente (Σx_i = T), non penalizzato
        - L'obiettivo è la pura somma delle distanze (senza il termine λ)

    Questo fornisce una soluzione di riferimento ottima, utile per valutare
    la qualità delle soluzioni euristiche/quantistiche.

    Parametri:
        instance: dizionario con distance_matrix, seed_indices, candidate_indices, T, n_points
        verbose: se True, stampa informazioni aggiuntive durante l'esecuzione

    Restituisce:
        dict con chiavi: solver, energy, feasible, time, selected

    Solleva:
        ImportError: se gurobipy non è disponibile

    Note:
        - energy è il valore dell'obiettivo MIP (solo distanze, senza penalità λ)
        - Il confronto diretto con SA/QA va fatto con cautela (energie diverse)
    """
    if not GUROBI_AVAILABLE:
        raise ImportError("gurobipy non disponibile")

    if verbose:
        print("  Costruzione e risoluzione del modello MIP...")

    d = instance["distance_matrix"]
    seed_idx = instance["seed_indices"]
    cand_idx = instance["candidate_indices"]
    T = instance["T"]
    n = instance["n_points"]

    model = gp.Model("QUACK_Expansion")
    model.Params.OutputFlag = 0  # Disabilita output verboso di Gurobi
    model.Params.TimeLimit = 300  # Timeout di 5 minuti

    # Variabili: x_i binaria per i candidati, x_i = 1 (fissata) per i seed
    x = {}
    for i in range(n):
        if i in seed_idx:
            x[i] = 1  # Il seed è sempre incluso (costante, non variabile)
        else:
            x[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}")

    # Obiettivo: minimizzare la somma delle distanze tra tutti i punti selezionati
    # Include le distanze seed-seed, seed-candidati e candidati-candidati
    obj = gp.quicksum(
        d[i][j] * x[i] * x[j]
        for i in range(n) for j in range(n) if i < j
    )
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Vincolo di cardinalità: selezionare esattamente T candidati
    model.addConstr(
        gp.quicksum(x[i] for i in cand_idx) == T,
        name="cardinality"
    )

    start = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - start

    # Conta i candidati selezionati nella soluzione
    selected = sum(
        1 for i in cand_idx
        if isinstance(x[i], gp.Var) and x[i].X > 0.5
    )

    # Valore obiettivo (inf se non ottimo)
    if model.Status == gp.GRB.OPTIMAL:
        obj_val = model.ObjVal
        feasible = True
    else:
        obj_val = float("inf")
        feasible = False

    if verbose:
        status_str = "OTTIMO" if feasible else f"STATUS={model.Status}"
        print(f"  Completato in {elapsed:.4f}s – {status_str}, Obj: {obj_val:.4f}, Selezionati: {selected}")

    return {
        "solver": "Gurobi",
        "energy": obj_val,
        "feasible": feasible,
        "time": elapsed,
        "selected": int(selected)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Funzione principale della demo.

    Orchestrazione:
        1. Parsing degli argomenti da riga di comando
        2. Lettura di λ e ID_Paper da lambda.csv
        3. Caricamento dell'istanza JSON
        4. Costruzione del BQM
        5. Esecuzione dei solver (SA sempre, QA e Gurobi se disponibili/richiesti)
        6. Stampa del riepilogo comparativo
    """
    # -------------------------------------------------------------------------
    # Parsing argomenti
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Demo QUACK – Espansione cluster bancario (QUBO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python demo_quack_banking.py --instance-id 96 --num-reads 1000
  python demo_quack_banking.py --instance-id 96 --verbose
  python demo_quack_banking.py --instance-id 96 --skip-dwave --skip-gurobi
        """
    )
    parser.add_argument(
        "--instance-id", type=int, required=True,
        help="ID dell'istanza (colonna ID in lambda.csv, es. 96, 97, ...)"
    )
    parser.add_argument(
        "--num-reads", type=int, default=1000,
        help="Numero di campionamenti per SA e QA (default: 1000)"
    )
    parser.add_argument(
        "--skip-dwave", action="store_true",
        help="Salta l'esecuzione su D-Wave anche se disponibile"
    )
    parser.add_argument(
        "--skip-gurobi", action="store_true",
        help="Salta l'esecuzione con Gurobi anche se disponibile"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stampa informazioni dettagliate durante l'esecuzione"
    )
    args = parser.parse_args()

    verbose = args.verbose

    # -------------------------------------------------------------------------
    # Setup percorsi
    # -------------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent
    lambda_csv = repo_root / "lambda.csv"

    # -------------------------------------------------------------------------
    # Lettura lambda.csv
    # -------------------------------------------------------------------------
    if not lambda_csv.exists():
        print(f"[ERRORE] File lambda.csv non trovato: {lambda_csv}")
        return

    try:
        lambda_val, id_paper = read_lambda_for_instance(lambda_csv, args.instance_id)
    except Exception as e:
        print(f"[ERRORE] Lettura lambda fallita: {e}")
        return

    # -------------------------------------------------------------------------
    # Caricamento istanza
    # -------------------------------------------------------------------------
    instance_file = repo_root / "istanze" / f"instance_{id_paper}.txt"
    if not instance_file.exists():
        print(f"[ERRORE] File istanza non trovato: {instance_file}")
        return

    instance, labels = load_instance_json(instance_file)

    # -------------------------------------------------------------------------
    # Costruzione BQM
    # -------------------------------------------------------------------------
    bqm = build_bqm(instance, lambda_val)

    # -------------------------------------------------------------------------
    # Header e riepilogo istanza
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Demo QUACK – Espansione del Cluster Bancario")
    print(f"{'='*60}")
    print(f"\nIstanza richiesta: ID = {args.instance_id}")
    print(f"Mapping: ID {args.instance_id} → ID_Paper {id_paper} → file instance_{id_paper}.txt")
    print(f"Lambda (λ): {lambda_val:.4f}")
    print(f"\nRiepilogo istanza:")
    print_instance_summary(instance, labels, verbose)
    print(f"\nVariabili QUBO (candidati): {len(instance['candidate_indices'])}")

    if verbose:
        print(f"Interazioni nel BQM: {len(bqm.quadratic)}")
        print(f"Num reads richiesti: {args.num_reads}")

    # -------------------------------------------------------------------------
    # Esecuzione solver
    # -------------------------------------------------------------------------
    results = []

    # --- Simulated Annealing (sempre eseguito) ---
    print_solver_header("Simulated Annealing (SA)", verbose)
    try:
        results.append(solve_sa(bqm, instance["T"], args.num_reads, verbose))
    except Exception as e:
        print(f"[WARN] SA fallito: {e}")
        if verbose:
            print(f"[INFO] Dettaglio errore SA: {type(e).__name__}: {e}")

    # --- D-Wave Quantum Annealer ---
    if not args.skip_dwave:
        print_solver_header("Quantum Annealing (D-Wave)", verbose)
        if not DWAVE_AVAILABLE:
            print("[WARN] D-Wave non disponibile (dwave.system non installato)")
            if verbose:
                print("[INFO] Per utilizzare D-Wave, installa dwave-ocean-sdk e configura il token API.")
        else:
            try:
                results.append(solve_dwave(bqm, instance["T"], args.num_reads, verbose))
            except Exception as e:
                print(f"[WARN] D-Wave fallito: {e}")
                if verbose:
                    print(f"[INFO] Verifica la configurazione D-Wave (token, connettività, quota).")

    # --- Gurobi ---
    if not args.skip_gurobi:
        print_solver_header("Gurobi (MIP esatto)", verbose)
        if not GUROBI_AVAILABLE:
            print("[WARN] Gurobi non disponibile (gurobipy non installato)")
            if verbose:
                print("[INFO] Per utilizzare Gurobi, installa gurobipy e attiva una licenza valida.")
        else:
            try:
                results.append(solve_gurobi(instance, verbose))
            except Exception as e:
                print(f"[WARN] Gurobi fallito: {e}")
                if verbose:
                    print(f"[INFO] Verifica la licenza Gurobi e i limiti del modello.")

    # -------------------------------------------------------------------------
    # Tabella riepilogativa
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Riepilogo risultati")
    print(f"{'='*60}\n")

    if not results:
        print("[WARN] Nessun solver ha prodotto risultati.")
        return

    print(f"{'Solver':<10} {'Energy':>12} {'Feasible':>10} {'Time[s]':>10} {'Selected':>10}")
    print("-" * 54)
    for r in results:
        feas_str = "Sì" if r["feasible"] else "No"
        print(f"{r['solver']:<10} {r['energy']:>12.4f} {feas_str:>10} {r['time']:>10.4f} {r['selected']:>10}")

    # -------------------------------------------------------------------------
    # Note interpretative
    # -------------------------------------------------------------------------
    print("\n--- Note ---")
    print("• Energy SA/QA: energia del BQM, include la penalità λ(Σx-T)²")
    print("• Energy Gurobi: pura somma delle distanze (vincolo imposto esattamente)")
    print("• Feasible: Sì se il numero di punti selezionati è esattamente T")

    # -------------------------------------------------------------------------
    # Analisi gap (solo in verbose e se Gurobi è presente)
    # -------------------------------------------------------------------------
    print_gap_analysis(results, verbose)

    print()  # Riga finale vuota per leggibilità


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
