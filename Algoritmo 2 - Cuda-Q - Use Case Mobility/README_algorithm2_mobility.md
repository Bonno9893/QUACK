# Progetto QUACK – Caso d'Uso Mobility (Algoritmo 2, QAOA con CUDA-Q)

Questa cartella contiene l'implementazione dell'**Algoritmo 2** del progetto QUACK, dedicato al **clustering binario (K=2)** su dataset di mobilità e istanze sintetiche "poligono".

Il problema di clustering viene mappato su un'**Hamiltoniana di tipo Max-Cut/Ising** e risolto con il **Quantum Approximate Optimization Algorithm (QAOA)** implementato in **CUDA-Q**. Per confronto, sono disponibili anche solver classici: **Simulated Annealing (SA)** e **Gurobi** (solver esatto per programmazione quadratica binaria).

Questo caso d'uso corrisponde allo **Use Case Mobility** descritto nei report WP2 (dataset) e WP4 (benchmark QAOA) del progetto QUACK.

> **Se vuoi provare subito la demo con un esempio già pronto, vai alla sezione [Come eseguire la demo (Google Colab)](#6-come-eseguire-la-demo-google-colab).**

---

## Indice

1. [Struttura della cartella](#1-struttura-della-cartella)
2. [Formato delle istanze](#2-formato-delle-istanze-istances_k2pkl)
3. [Idea dell'Algoritmo 2](#3-idea-dellalgoritmo-2-k2-max-cut-e-qaoa)
4. [File principali](#4-file-principali)
5. [Come eseguire la demo (Google Colab)](#5-come-eseguire-la-demo-google-colab)
6. [Esecuzioni e analisi avanzate](#6-esecuzioni-e-analisi-avanzate)
7. [Limitazioni e prerequisiti](#7-limitazioni-e-prerequisiti)
8. [Riferimenti al progetto QUACK](#8-riferimenti-al-progetto-quack)

---

## 1. Struttura della cartella

```text
algorithm2_mobility/
├── README.md
├── demo_quack_mobility.ipynb           # Demo interattiva (Colab)
├── CudaQ_clustering_k2_implementation.py   # Modulo core QAOA
├── Analisi_QAOA_GPU_CPU.py             # Script di analisi prestazionale
├── istances_k2/
│   ├── istanza_0.pkl
│   ├── istanza_1.pkl
│   ├── istanza_2.pkl
│   ├── ...
│   └── VISUALIZZAZIONI/                # Grafici delle istanze
└── Istances/                           # Eventuali istanze aggiuntive
```

**Contenuto principale:**

- **`istances_k2/`**: contiene le istanze K=2 pronte all'uso. Le istanze sono sintetiche ("poligono") e servono per benchmarking e validazione dell'algoritmo QAOA.
- **`demo_quack_mobility.ipynb`**: notebook Jupyter/Colab per eseguire una demo completa su una singola istanza.
- **`CudaQ_clustering_k2_implementation.py`**: modulo Python con l'implementazione core di QAOA, incluse le funzioni per costruire l'Hamiltoniana, eseguire VQE e campionare.
- **`Analisi_QAOA_GPU_CPU.py`**: script per analisi comparative GPU vs CPU, con generazione di grafici accademici.

---

## 2. Formato delle istanze (`istances_k2/*.pkl`)

Ogni file `.pkl` è un dizionario Python serializzato con pickle, contenente:

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `distance_matrix` | `np.ndarray` (n×n) | Matrice simmetrica delle distanze tra punti |
| `y_true` | `np.ndarray` (n,) | Etichette ground-truth per il clustering (0 o 1) |
| `info` | `dict` | Metadati dell'istanza |
| `X` | `np.ndarray` (n×2) | Coordinate dei punti nel piano (opzionale) |
| `distance_matrix_normalized` | `np.ndarray` (n×n) | Matrice delle distanze normalizzata (opzionale) |

**Struttura del campo `info`:**

```python
{
    "n_samples": int,          # Numero di punti
    "n_clusters": int,         # Numero di cluster (sempre 2 per K=2)
    "cluster_std": float,      # Deviazione standard dei cluster
    "bilanciato": bool,        # Se i cluster sono bilanciati
    "points_per_cluster": int, # Punti per cluster
    "ari": float               # ARI della soluzione ottima (se noto)
}
```

**Esempio**: `istanza_0.pkl` contiene 4 punti in 2D con 2 cluster, come mostrato nella visualizzazione.

---

## 3. Idea dell'Algoritmo 2 (K=2, Max-Cut e QAOA)

Per il clustering binario (K=2), ogni punto viene mappato su un **qubit**:
- Lo stato `|0⟩` rappresenta l'appartenenza al cluster 0
- Lo stato `|1⟩` rappresenta l'appartenenza al cluster 1

La matrice di distanza viene trasformata in un'**Hamiltoniana di costo** di tipo Max-Cut/Ising. La funzione `hamiltonian_k2` in `CudaQ_clustering_k2_implementation.py` costruisce questa Hamiltoniana come:

```
H = Σ_{i<j} (d_ij / 2) * (I + Z_i Z_j)
```

dove `d_ij` è la distanza tra i punti i e j. Minimizzare questa Hamiltoniana equivale a minimizzare la **distanza intra-cluster**.

**QAOA** alterna due tipi di operazioni:
1. **Cost layer**: evoluzione secondo l'Hamiltoniana di costo con parametro γ
2. **Mixer layer**: rotazioni RX con parametro β per esplorare lo spazio delle soluzioni

Il kernel QAOA (`kernel_qaoa`) implementa p layer di queste operazioni. I parametri (γ₁, β₁, ..., γₚ, βₚ) vengono ottimizzati tramite **VQE** (Variational Quantum Eigensolver) usando l'ottimizzatore Nelder-Mead.

La qualità del clustering viene valutata con:
- **Costo**: somma delle distanze tra punti nello stesso cluster (più basso = migliore)
- **ARI (Adjusted Rand Index)**: confronto con le etichette ground-truth (1.0 = perfetto)

---

## 4. File principali

### 4.1 `demo_quack_mobility.ipynb`

È il **punto di ingresso consigliato** per chi vuole provare velocemente QAOA su un'istanza.

Il notebook:
1. Installa e configura **CUDA-Q** e le dipendenze (`cudaq`, `scikit-learn`, `gurobipy`)
2. Permette di selezionare un'istanza `.pkl` dalla cartella `istances_k2/`
3. Definisce e esegue tre solver:
   - **QAOA (CUDA-Q)**: con parametri configurabili (backend, p_layers, shots, seed)
   - **Simulated Annealing**: metaeuristica classica semplice
   - **Gurobi**: solver esatto (opzionale, richiede licenza)
4. Mostra una **tabella comparativa** con costo, ARI e tempo di esecuzione

### 4.2 `CudaQ_clustering_k2_implementation.py`

Modulo Python contenente l'implementazione core. **Non è pensato come demo** ma come libreria che altri script richiamano.

**Funzioni principali:**

| Funzione | Descrizione |
|----------|-------------|
| `hamiltonian_k2(distance_matrix, scale)` | Costruisce l'Hamiltoniana di costo |
| `kernel_qaoa(n_qubits, p, edges_src, edges_tgt, weights, thetas)` | Kernel QAOA con p layer |
| `pick_backend(preferred)` | Seleziona il backend CUDA-Q (nvidia, qpp-cpu, ecc.) |
| `shots_for_n(n, base)` | Calcola il numero di shots ottimale per n qubit |
| `calculate_clustering_cost(distance_matrix, bitstring)` | Calcola il costo di un clustering |
| `run_qaoa(...)` | Funzione principale che esegue VQE + sampling |

**`run_qaoa` restituisce:**
- `best_cost`: costo migliore trovato
- `prob_best`: probabilità di misurare la soluzione migliore
- `ari`: Adjusted Rand Index vs ground-truth
- `theta_used`: parametri QAOA ottimizzati
- `results_dict`: dizionario {bitstring: (count, energy, cost)}
- `times`: breakdown temporale (t_vqe, t_quantum, t_classic, t_sampling, ecc.)
- `metrics`: metriche aggiuntive (n_unique_bitstrings, ecc.)

### 4.3 `Analisi_QAOA_GPU_CPU.py`

Script per **analisi prestazionale avanzata** dei risultati QAOA.

**Funzionalità:**
- Carica risultati da file CSV (GPU, CPU, parziali) ed Excel (D-Wave)
- Genera grafici accademici organizzati in sottocartelle:
  - Distribuzione dimensioni istanze
  - Convergenza QAOA (prob_opt vs layers)
  - Confronto tempi GPU vs CPU
  - Confronto con D-Wave
  - Copertura dello spazio delle soluzioni
  - Analisi early stopping
  - Time-to-Solution (TTS)
- Produce report testuale con statistiche dettagliate

**Uso:**
```bash
python Analisi_QAOA_GPU_CPU.py \
    --gpu risultati_gpu.csv \
    --cpu risultati_cpu.csv \
    --partial risultati_partial.csv \
    --dwave risultati_dwave.xlsx
```

---

## 5. Come eseguire la demo (Google Colab)

### Passo 1: Preparare i file

Carica su Google Colab (o Google Drive):
- `demo_quack_mobility.ipynb`
- `CudaQ_clustering_k2_implementation.py`
- La cartella `istances_k2/` con almeno un file `.pkl`

### Passo 2: Aprire il notebook

1. Vai su [Google Colab](https://colab.research.google.com/)
2. Carica `demo_quack_mobility.ipynb`
3. **Importante**: Abilita il runtime GPU da `Runtime > Change runtime type > GPU`

### Passo 3: Eseguire il setup

Esegui le prime celle per installare le dipendenze:
```python
!pip install -q cudaq scikit-learn dimod
!pip install -q gurobipy  # opzionale
```

### Passo 4: Impostare il percorso delle istanze

Modifica la variabile `INSTANCES_DIR` nella cella apposita:
```python
# Se hai caricato direttamente su Colab:
INSTANCES_DIR = Path("/content/istances_k2")

# Se usi Google Drive:
INSTANCES_DIR = Path("/content/drive/MyDrive/QUACK/istances_k2")
```

### Passo 5: Eseguire tutte le celle

Esegui le celle in ordine dall'alto verso il basso. L'output finale mostrerà:

```
========== QUACK Use Case 2 - K=2 Clustering DEMO ==========
Instance: istanza_0.pkl
n_points: 4
============================================================

   solver      cost       ARI    time_s
0  Gurobi  XXX.XXXX  X.XXXX    X.XXX
1      SA  XXX.XXXX  X.XXXX    X.XXX
2    QAOA  XXX.XXXX  X.XXXX    X.XXX

QAOA backend: nvidia, shots: XXXX
QAOA prob_best (model): X.XXX, empirical: X.XXX
```

---

## 6. Esecuzioni e analisi avanzate

### Modificare i parametri QAOA

Nel notebook, puoi modificare le variabili di configurazione:

```python
backend_name = "nvidia"   # oppure "qpp-cpu" per CPU
p_layers = 3              # numero di layer QAOA (1-10)
seed = 27                 # seed per riproducibilità
manual_shots = None       # None = auto, oppure un intero
```

**Suggerimenti:**
- Più layer (`p_layers`) = circuito più profondo, potenzialmente risultati migliori ma tempo maggiore
- Backend `nvidia` è più veloce ma richiede GPU; `qpp-cpu` funziona ovunque
- Aumentare `shots` migliora la stima delle probabilità ma aumenta il tempo

### Testare più istanze

Per cambiare istanza:
```python
instances = list_available_instances()
instance_path = instances[2]  # Seleziona la terza istanza
```

### Eseguire l'analisi GPU vs CPU

Se hai eseguito benchmark estesi e salvato i risultati in CSV:
```bash
python Analisi_QAOA_GPU_CPU.py \
    --gpu qaoa_results_gpu.csv \
    --cpu qaoa_results_cpu.csv \
    --partial qaoa_results_partial.csv
```

L'output viene salvato in una cartella `qaoa_analysis_academic_YYYYMMDD_HHMMSS/` con grafici e report.

---

## 7. Limitazioni e prerequisiti

### Prerequisiti tecnici

**Per Google Colab:**
- Runtime con GPU abilitato (consigliato T4 o superiore)
- Installazione di `cudaq` tramite pip

**Per esecuzione locale:**
- Linux (CUDA-Q ha supporto limitato su Windows/Mac)
- Python 3.8+
- GPU NVIDIA con driver CUDA compatibili (per backend `nvidia`)
- Oppure esecuzione CPU-only con backend `qpp-cpu`

### Limitazioni

- **Dataset Mobility originale**: i dati reali del dataset Mobility non sono inclusi in questa cartella per motivi di privacy. Le istanze fornite sono sintetiche ("poligono") per scopi di test e validazione.
- **Scalabilità**: QAOA su simulatore classico scala esponenzialmente con il numero di qubit. Istanze oltre ~20 punti richiedono tempi significativi.
- **Gurobi**: richiede licenza (gratuita per uso accademico). Se non disponibile, il solver Gurobi viene automaticamente disabilitato.
- **Path hardcoded**: alcuni percorsi potrebbero essere specifici per l'ambiente di sviluppo originale e richiedere adattamento.

---

## 8. Riferimenti al progetto QUACK

Questo codice è stato sviluppato nel contesto del progetto **QUACK – Quantum Customer Knowledge**, finanziato dalla Regione Emilia-Romagna nell'ambito del programma POR-FESR 2021-2027.

L'Algoritmo 2 (Mobility) e le relative istanze sono descritti nei report interni:
- **WP2**: definizione del dataset Mobility e preprocessing
- **WP4**: benchmark QAOA su CUDA-Q, confronto GPU/CPU/D-Wave

Per ulteriori informazioni sul progetto QUACK, consultare il README principale del repository.

---

*Ultimo aggiornamento: Dicembre 2024*
