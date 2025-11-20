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

## Dataset originale

Il caso d’uso bancario si basa sul dataset pubblico **“New Marketing Campaign”**, disponibile su Kaggle:  
<https://www.kaggle.com/datasets/mikejimenez24/new-marketing-campaign>.

Il dataset contiene oltre 300.000 clienti e descrive, per ciascuno, tre gruppi principali di informazioni:  
- **variabili economiche** (es. reddito annuale, importi spesi in diverse categorie di prodotto come vino, carne, pesce, dolci, beni di lusso);  
- **variabili di comportamento d’acquisto** (es. numero di acquisti online, su catalogo, in negozio, visite al sito, utilizzo di sconti);  
- **variabili demografiche** (es. età, livello di istruzione, stato civile, composizione del nucleo familiare). :contentReference[oaicite:0]{index=0}  

Nel progetto QUACK, il dataset è stato preprocessato rimuovendo le variabili non direttamente legate al comportamento di spesa (es. informazioni socio-demografiche, indicatori di campagne marketing) e mantenendo solo le feature che descrivono **quanto** e **come** i clienti acquistano. Le variabili quantitative sono state standardizzate (media zero, varianza unitaria) per rendere coerente il calcolo delle distanze.

## Costruzione delle istanze

A partire dal dataset preprocessato è stata definita una pipeline in due fasi: **segmentazione iniziale** e **generazione delle istanze**. :contentReference[oaicite:1]{index=1}  

1. **Clustering e selezione dei clienti rappresentativi**  
   - Si applica un algoritmo di clustering (K-Means) sul dataset standardizzato, scegliendo **2 cluster** come numero ottimale.  
   - Per ciascun cluster si selezionano i punti più rappresentativi, ossia quelli più vicini al centroide nello spazio delle feature.  
   - Da questi si costruiscono tre “pool” di riferimento: **Best 400**, **Best 1000** e **Best 2000** punti, che fungono da base per tutte le istanze successive.

2. **Definizione delle istanze di test**  
   A partire dai pool, vengono generate istanze sperimentali che rappresentano diversi scenari di difficoltà. Ogni istanza è caratterizzata da:
   - un **cluster iniziale** I_0, composto da un sottoinsieme di clienti “compatibili” tra loro (seed su cui l’algoritmo deve lavorare);  
   - un insieme di **punti candidati all’espansione** C, estratti dal pool e potenzialmente aggiungibili a I_0;  
   - una **combinazione di parametri strutturali**, che controllano:
     - la dimensione relativa di I_0 rispetto ai punti totali (ad esempio, configurazioni con I_0 pari al 25%, 50% o 75% del totale);  
     - la **percentuale di punti compatibili** dentro C (es. 20%, 40%, 50%, 80%), che introduce un diverso livello di “rumore” nel problema;  
     - il **numero complessivo di punti** nell’istanza (es. configurazioni con 4, 8, 16, 32 punti), scelto anche in funzione dei vincoli del quantum annealer.

3. **Struttura logica di una istanza**

In termini concettuali, ogni istanza contiene almeno:

- l’elenco dei punti coinvolti (clienti selezionati dal pool) e i relativi **indici**;  
- l’indicazione di quali punti appartengono al **cluster seed** I_0 e quali sono in C;  
- il valore target T, cioè quanti nuovi punti il modello deve aggiungere a I_0;  
- la **matrice di distanza** d_{ij} tra tutti i punti dell’istanza (estratta dalla matrice globale del pool);  
- alcuni **metadati strutturali** (dimensioni del pool, configurazione di rumore, scenario di difficoltà).

In questa repository le istanze sono fornite in formato testuale (`.txt`), ma mantenengono la stessa struttura logica (seed I_0, candidati C, parametri, matrice di distanza) in una forma più facilmente consultabile e riutilizzabile.


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
