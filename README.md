[README.md](https://github.com/user-attachments/files/23634444/README.md)
# Progetto QUACK - Caso d'Uso Bancario con D-Wave Quantum Annealer

## Panoramica del Progetto

Il progetto QUACK (QUAntum Clustering for Knowledge) esplora l'applicazione delle tecniche di calcolo quantistico ai problemi di clustering, con un focus specifico sulla segmentazione dei clienti nel settore bancario. Questo repository contiene l'implementazione dell'Algoritmo 1 (Espansione del Cluster) del progetto, utilizzando la tecnologia di quantum annealing di D-Wave, insieme a benchmark classici usando Gurobi e Simulated Annealing.

### Obiettivi Principali

- **Ottimizzazione Quantistica**: Sfruttare il quantum annealer di D-Wave per risolvere problemi di clustering vincolato
- **Analisi Comparativa**: Confrontare le soluzioni quantistiche con i metodi classici (Gurobi, Simulated Annealing)
- **Applicazione Reale**: Applicare il clustering quantistico alla segmentazione dei clienti bancari basata sui pattern di spesa
- **Ottimizzazione dei Parametri**: Implementare l'ottimizzazione adattiva del parametro λ (lambda) per le formulazioni QUBO

## Struttura del Repository

```
QUACK-Banking-DWave-Clustering/
│
├── README.md                           # Questo file
├── requirements.txt                    # Dipendenze Python
├── config.yaml                        # Impostazioni di configurazione
│
├── dati/                              # Dati e istanze
│   ├── istanze/                       # Istanze di test generate
│   ├── grezzi/                        # Dataset bancario originale
│   └── processati/                    # Dati preprocessati
│
├── src/                               # Codice sorgente
│   ├── generazione_istanze/         # Script per creazione istanze
│   │   ├── crea_istanze_bancarie.py
│   │
│   ├── ottimizzazione/                # Algoritmi di ottimizzazione core
│   │   ├── ottimizzatore_lambda.py   # Ottimizzazione parametro lambda
│   │   ├── formulazione_qubo.py      # Costruzione modello QUBO
│   │
│   ├── risolutori/                    # Implementazioni dei diversi solver
│   │   ├── risolutore_dwave.py       # Quantum annealing D-Wave
│   │   ├── risolutore_gurobi.py      # (DA AGGIUNGERE) Solver esatto Gurobi
│   │   └── simulated_annealing.py    # (DA AGGIUNGERE) Implementazione SA classica
│
├── script/                            # Script di esecuzione
│   ├── esegui_pipeline_completa.py  # Pipeline principale

```

## Avvio Rapido

### Prerequisiti

1. **Python 3.8+** installato
2. **D-Wave Ocean SDK** account e token API (per esecuzione quantistica)
3. **Licenza Gurobi** (per benchmark classico)

### Installazione

```bash
# Clona il repository
git clone https://github.com/tuousername/QUACK-Banking-DWave-Clustering.git
cd QUACK-Banking-DWave-Clustering

# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### Configurazione

1. Copia il template di configurazione:
```bash
cp config.yaml.template config.yaml
```

2. Modifica `config.yaml` con le tue credenziali:
```yaml
dwave:
  token_api: "IL_TUO_TOKEN_DWAVE"
  risolutore: "Advantage_system4.1"  # o il tuo solver preferito

gurobi:
  percorso_licenza: "/percorso/a/gurobi.lic"

percorsi:
  cartella_dati: "./dati"
  cartella_risultati: "./risultati"
```

### Esecuzione della Pipeline Completa

```bash
python script/esegui_pipeline_completa.py --config config.yaml
```

Questo comando:
1. Genera o carica le istanze di test
2. Ottimizza i parametri λ usando Simulated Annealing
3. Risolve usando il quantum annealer D-Wave
4. Confronta con i metodi classici
5. Genera report di performance e visualizzazioni

## Descrizione dell'Algoritmo

### Algoritmo 1: Espansione del Cluster

L'algoritmo di espansione del cluster affronta il problema di aggiungere esattamente T nuovi punti a un seed di cluster esistente, minimizzando le distanze intra-cluster mentre rispetta i vincoli di cardinalità.

#### Formulazione QUBO

Il problema è formulato come Ottimizzazione Binaria Quadratica Non Vincolata (QUBO):

```
min Σ(i,j) d_ij * x_i * x_j + λ₂ * (Σx_i - T)²
```

Dove:
- `d_ij`: Distanza tra i punti i e j
- `x_i`: Variabile binaria (1 se il punto i è selezionato, 0 altrimenti)
- `T`: Numero target di punti da selezionare
- `λ₂`: Parametro di penalità per il vincolo di cardinalità

### Ottimizzazione dei Parametri

Il parametro λ₂ è cruciale per la qualità della soluzione ed è ottimizzato attraverso:
1. **Ricerca Adattiva su Griglia**: Test di valori crescenti di λ₂
2. **Verifica di Fattibilità**: Assicura che esattamente T punti siano selezionati
3. **Consistenza Geometrica**: Valutazione della compattezza del cluster
4. **Cross-validazione**: Usando SA e Gurobi come risolutori di riferimento

## Metriche di Performance

Il framework valuta le soluzioni usando multiple metriche:

- **Indice di Rand Aggiustato (ARI)**: Misura l'accordo del clustering con il ground truth
- **Distanza Intra-cluster**: Distanza totale all'interno dei cluster
- **Tasso di Fattibilità**: Percentuale di soluzioni valide
- **Tempo di Elaborazione Quantistico**: Tempo di accesso QPU
- **Overhead di Embedding**: Tempo per il minor embedding sull'hardware quantistico

## Esempi di Utilizzo

### Creazione di Istanze Bancarie

```python
from src.generazione_istanze.crea_istanze_bancarie import GeneratoreIstanzeBancarie

generatore = GeneratoreIstanzeBancarie(
    n_clienti=16,
    n_caratteristiche=2,
    n_cluster=2
)

istanze = generatore.genera_istanze(
    n_istanze=10,
    dimensione_seed=50,
    dimensione_espansione=20
)
```

### Ottimizzazione del Parametro Lambda

```python
from src.ottimizzazione.ottimizzatore_lambda import OttimizzatoreLambda

ottimizzatore = OttimizzatoreLambda(
    range_lambda=(0.1, 10.0),
    passo=0.1
)

lambda_ottimale = ottimizzatore.ottimizza(
    istanza=istanza,
    risolutore='simulated_annealing',
    max_iterazioni=100
)
```

### Risoluzione con D-Wave

```python
from src.risolutori.risolutore_dwave import RisolutoreDWave

risolutore = RisolutoreDWave(token_api="IL_TUO_TOKEN")

soluzione = risolutore.risolvi(
    matrice_distanze=matrice_distanze,
    n_selezionare=20,
    penalita_lambda=lambda_ottimale,
    num_letture=1000
)
```

## Riepilogo dei Risultati

## Team di Sviluppo

Benedetta Ferrari, Mirko Mucciarini, Filippo Bonafè

## Informazioni Aggiuntive

https://www.linkedin.com/company/quack-project

https://www.re-lab.it/projects/quack


## Pubblicazioni

Ferrari, Benedetta, et al. "Lookalike Clustering for Customer
Segmentation: a Comparative Study of Quantum Annealing and
Classical Algorithms." Proceedings of the Genetic and 
Evolutionary Computation Conference Companion. 2025.

---
*Parte del Progetto QUACK (QUAntum Clustering for Knowledge) - Avanzamento delle applicazioni di calcolo quantistico in scenari di clustering del mondo reale*
