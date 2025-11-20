[README.md](https://github.com/user-attachments/files/23634444/README.md)
# Progetto QUACK - Caso d'Uso Bancario (Algoritmo 1) con D-Wave Quantum Annealer

## Panoramica

Questo repository contiene il codice principale utilizzato nel progetto **QUACK – QUAntum Clustering for Knowledge** per il **caso d’uso bancario** basato sull’**Algoritmo 1 (espansione del cluster)**.  Il codice mostra come formulare il problema in forma **QUBO** e come risolverlo usando sia il **quantum annealer di D-Wave** sia due riferimenti classici: **Gurobi** (solver esatto) e **Simulated Annealing** (metaeuristica).

L’obiettivo è documentare in modo riproducibile la pipeline sperimentale usata nel progetto, in modo che chiunque nel gruppo di ricerca possa:

- capire rapidamente che problema viene risolto,
- vedere come sono state generate le istanze,
- riutilizzare gli script per nuovi esperimenti.


## Struttura del Repository

```
QUACK/
├── README.md
├── requirements.txt
│
├── crea_istanze_bancarie.py   # Generazione delle istanze bancarie
├── ottimizzatore_lambda.py    # Ottimizzazione del parametro lambda
├── risolutore_dwave.py        # Risoluzione QUBO su D-Wave
├── risolutore_gurobi.py       # Risoluzione QUBO con Gurobi
├── risolutore_simulated_annealing.py     # Risoluzione QUBO con Simulated Annealing
│
├── istanze/
│   ├── istanza_1.txt
│   ├── istanza_2.txt
│   └── ...
│
└── lambda.csv
```
dove i file .txt rappresentano le istanze di test già pronte all'uso.

- **crea_istanze_bancarie.py**  
  Genera le istanze del problema di espansione del cluster a partire dal dataset bancario originale (non incluso nel repository). Nella versione originale del progetto le istanze venivano salvate in uno o più file `.pkl`, utilizzati dagli altri script per gli esperimenti. In questo repository sono inoltre presenti file di istanza in formato testuale (`.txt`), creati ad hoc per il paper dedicato all’algoritmo e corrispondenti alle stesse istanze utilizzate nelle analisi sperimentali. Il file `crea_istanze_bancarie.py` è incluso a scopo di consultazione, per documentare il processo di generazione delle istanze originali, che sono pienamente equivalenti alle versioni in formato `.txt` presenti in questo repository.

- **ottimizzatore_lambda.py**  
  Esegue la procedura di ottimizzazione del parametro lambda nella formulazione QUBO. Per ciascuna istanza testa diversi valori di lambda, lancia il solver interno e valuta la qualità/fattibilità delle soluzioni per selezionare un valore di lambda appropriato.

- **risolutore_dwave.py**  
  Risolve le istanze QUBO utilizzando il quantum annealer di **D-Wave** tramite l’Ocean SDK. Si occupa di costruire l’input per D-Wave, impostare i parametri principali di esecuzione (ad esempio numero di read) e raccogliere i campioni restituiti dall’hardware.

- **risolutore_gurobi.py**  
  Implementa la stessa formulazione del problema in un modello **Gurobi** e lo risolve come benchmark classico esatto. Fornisce una soluzione di riferimento per confrontare i risultati ottenuti con D-Wave e con gli approcci euristici.

- **risolutore_simulated_annealing.py**  
  Contiene un risolutore classico basato su **Simulated Annealing** per il QUBO. Dati i parametri di un’istanza e le impostazioni dell’algoritmo, produce una soluzione approssimata da confrontare con le soluzioni quantistiche e con Gurobi.

- **lambda.csv**  
    Il file `lambda.csv` contiene i valori di \(\lambda\) ottimizzati per le istanze in formato `.txt` presenti in questo repository.  
    Per ogni istanza sono riportati:
    
    - un identificativo coerente con le etichette utilizzate nel paper e nei nomi delle istanze;
    - alcuni metadati (ad esempio bacino, livello di rumore, dimensioni dell’istanza);
    - il valore di \(\lambda\) scelto per gli esperimenti, nella colonna `LAMBDA`.
    
    Questo file permette di **replicare direttamente le configurazioni usate nel paper**, senza dover rilanciare la procedura di ottimizzazione in `ottimizzatore_lambda.py`.

## Avvio Rapido

### Prerequisiti

1. **Python 3.8+** installato
2. **D-Wave Ocean SDK** account e token API (per esecuzione quantistica)
3. **Licenza Gurobi** (per benchmark classico)

### Installazione

```bash
# Clona il repository
git clone https://github.com/tuousername/QUACK.git
cd QUACK

# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

### Configurazione

### Esecuzione

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

### Istanze Bancarie



### Ottimizzazione del Parametro Lambda


### Risoluzione con D-Wave


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
