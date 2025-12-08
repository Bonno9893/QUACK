[README_algorithm1_banking.md](https://github.com/user-attachments/files/23634444/README.md)
### README per Algoritmo 1
# Progetto QUACK - Caso d'Uso Bancario (Algoritmo 1) con D-Wave Quantum Annealer

## Panoramica

Questo repository contiene il codice principale utilizzato nel progetto **QUACK – QUAntum Clustering for Knowledge** per il **caso d’uso bancario** basato sull’**Algoritmo 1 (espansione del cluster)**. Il codice mostra come formulare il problema in forma **QUBO** e come risolverlo usando sia il **quantum annealer di D-Wave** sia due riferimenti classici: **Gurobi** (solver esatto) e **Simulated Annealing** (metaeuristica).

Se vuoi **eseguire subito una demo end-to-end** dell’algoritmo su una istanza di esempio, vai direttamente alla sezione [Demo end-to-end (uso rapido)](#demo-end-to-end-uso-rapido).

L’obiettivo di questa cartella è documentare in modo riproducibile la pipeline sperimentale usata nel progetto, in modo che chiunque nel gruppo di ricerca possa:

- capire rapidamente che problema viene risolto,
- vedere come sono state generate le istanze,
- riutilizzare gli script per nuovi esperimenti.

## Struttura del Repository

La struttura logica di questa cartella (dedicata all’Algoritmo 1 – caso d’uso bancario) è la seguente:

```text
Algoritmo 1 - D-Wave - Use Case Bancario/
├── README.md
│
├── crea_istanze_bancarie.py          # Generazione delle istanze bancarie (versione originale)
├── ottimizzatore_lambda.py           # Ottimizzazione del parametro lambda
├── risolutore_dwave.py               # Risoluzione QUBO su D-Wave
├── risolutore_gurobi.py              # Risoluzione QUBO con Gurobi
├── risolutore_simulated_annealing.py # Risoluzione QUBO con Simulated Annealing
├── demo_quack_banking.py             # Script di demo end-to-end dell'Algoritmo 1
│
├── istanze/
│   ├── instance_0.txt
│   ├── instance_1.txt
│   ├── ...
│   └── instance_N.txt
│
└── lambda.csv
````

dove i file `.txt` rappresentano le istanze di test già pronte all'uso.

* **crea_istanze_bancarie.py**
  Genera le istanze del problema di espansione del cluster a partire dal dataset bancario originale (non incluso nel repository). Nella versione originale del progetto le istanze venivano salvate in uno o più file `.pkl`, utilizzati dagli altri script per gli esperimenti. In questa cartella sono inoltre presenti file di istanza in formato testuale (`.txt`), creati ad hoc per il paper dedicato all’algoritmo e corrispondenti alle stesse istanze utilizzate nelle analisi sperimentali. Il file `crea_istanze_bancarie.py` è incluso a scopo di consultazione, per documentare il processo di generazione delle istanze originali, che sono pienamente equivalenti alle versioni in formato `.txt` presenti qui.

* **ottimizzatore_lambda.py**
  Esegue la procedura di ottimizzazione del parametro lambda nella formulazione QUBO. Per ciascuna istanza testa diversi valori di lambda, lancia il solver interno (tipicamente Simulated Annealing) e valuta la qualità/fattibilità delle soluzioni per selezionare un valore di lambda appropriato.

* **risolutore_dwave.py**
  Risolve le istanze QUBO utilizzando il quantum annealer di **D-Wave** tramite l’Ocean SDK. Si occupa di costruire l’input per D-Wave, impostare i parametri principali di esecuzione (ad esempio numero di read) e raccogliere i campioni restituiti dall’hardware.

* **risolutore_gurobi.py**
  Implementa la stessa formulazione del problema in un modello **Gurobi** e lo risolve come benchmark classico esatto. Fornisce una soluzione di riferimento per confrontare i risultati ottenuti con D-Wave e con gli approcci euristici.

* **risolutore_simulated_annealing.py**
  Contiene un risolutore classico basato su **Simulated Annealing** per il QUBO. Dati i parametri di un’istanza e le impostazioni dell’algoritmo, produce una soluzione approssimata da confrontare con le soluzioni quantistiche e con Gurobi.

* **demo_quack_banking.py**
  Script di **demo end-to-end** che mette insieme, in un unico flusso, caricamento di una istanza, costruzione del QUBO, chiamata al solver e valutazione del clustering ottenuto. È il punto di ingresso consigliato se si vuole vedere rapidamente l’algoritmo in azione su una istanza di esempio.

* **lambda.csv**
  Il file `lambda.csv` contiene i valori di (\lambda) ottimizzati per le istanze in formato `.txt` presenti in questa cartella.
  Per ogni istanza sono riportati:

  * un identificativo coerente con le etichette utilizzate nel paper e nei nomi delle istanze;
  * alcuni metadati (ad esempio bacino, livello di rumore, dimensioni dell’istanza);
  * il valore di (\lambda) scelto per gli esperimenti, nella colonna `LAMBDA`.

  Questo file permette di **replicare direttamente le configurazioni usate nel paper**, senza dover rilanciare la procedura di ottimizzazione in `ottimizzatore_lambda.py`.

## Demo end-to-end (uso rapido)

Questa sezione descrive come eseguire rapidamente una demo completa dell’Algoritmo 1 su una istanza di esempio, senza dover passare subito dai singoli script.

### Esecuzione della demo

Assumendo di essere nella root del repository principale (`QUACK/`), è possibile lanciare la demo con:

```bash
cd "Algoritmo 1 - D-Wave - Use Case Bancario"
python demo_quack_banking.py
```

Oppure, se stai già lavorando dentro la cartella dell’Algoritmo 1:

```bash
python demo_quack_banking.py
```

Prerequisiti minimi:

* ambiente Python configurato con le dipendenze necessarie (vedi sezione *Avvio Rapido* nel README principale del progetto, oppure un `requirements.txt` condiviso),
* istanze `.txt` presenti nella cartella `istanze/`,
* file `lambda.csv` coerente con i nomi delle istanze, se la demo legge λ da lì,
* eventuali credenziali D-Wave configurate, se la demo prevede la chiamata effettiva al quantum annealer.

### Cosa fa la demo, passo per passo

In termini logici, `demo_quack_banking.py` esegue una pipeline di questo tipo:

1. **Selezione e caricamento di una istanza**
   La demo sceglie una delle istanze presenti in `istanze/` (ad esempio `instance_0.txt` o un indice configurato nello script) e ne legge:

   * la matrice di distanza o similarità tra i clienti considerati,
   * i parametri strutturali dell’istanza (numero di punti, T, eventuali metadati).

2. **Impostazione o lettura del parametro λ**
   Il parametro di penalità (\lambda) viene:

   * letto dal file `lambda.csv`, se la demo è configurata per usare i valori pre-calibrati per ogni istanza, oppure
   * fissato internamente nello script, per scopi di test.

3. **Costruzione del QUBO**
   A partire dall’istanza e dal valore di λ, la demo costruisce la matrice QUBO (Q) che rappresenta il problema di espansione del cluster:

   * il termine di costo principale misura la coesione del cluster (distanze intra-cluster),
   * il termine di penalità, pesato da λ, impone che il numero di punti selezionati sia esattamente (T).

4. **Chiamata al solver**
   La demo invoca uno dei solver disponibili (tipicamente configurato tramite una variabile o una sezione di codice):

   * **D-Wave** (quantum annealer),
   * oppure **Simulated Annealing**,
   * oppure **Gurobi**, se si vuole utilizzare la versione MILP/MIP del modello.

   Lo script:

   * prepara l’input nella forma richiesta dal solver (es. `BinaryQuadraticModel` per D-Wave/SA),
   * esegue la chiamata con i parametri scelti (numero di read, tempo di esecuzione, ecc.),
   * raccoglie il bitstring migliore (o i migliori) trovati.

5. **Decodifica del bitstring in clustering**
   Il bitstring restituito viene reinterpretato come vettore di variabili binarie di assegnazione:

   * da qui si ricava quali candidati sono stati aggiunti al seed del cluster,
   * si verifica la **fattibilità** rispetto al vincolo sul numero di punti selezionati ((T)),
   * si costruisce la partizione finale dei clienti dell’istanza.

6. **Calcolo e stampa delle metriche**
   Infine, la demo calcola alcune metriche di qualità, ad esempio:

   * valore della funzione obiettivo QUBO per la soluzione trovata,
   * misure interne di coesione del cluster (es. somma delle distanze intra-cluster),
   * eventuali metriche esterne (es. ARI) se l’istanza include etichette di riferimento.

   I risultati vengono tipicamente stampati a schermo e, in alcune versioni, possono essere salvati su file CSV per analisi successive.

### Come usare la demo per fare esperimenti

Per adattare la demo ai tuoi test, puoi:

* cambiare quale istanza viene caricata (ad esempio modificando l’indice o il nome del file all’inizio di `demo_quack_banking.py`),
* sostituire il solver di default (es. passare da SA a D-Wave o Gurobi modificando una variabile o il blocco `if __name__ == "__main__"`),
* modificare λ (sia nel codice, sia nel file `lambda.csv`),
* modificare i parametri del solver (numero di read, tempi, schedule, ecc.) per studiarne l’impatto sulle soluzioni.

La demo è pensata come **entry point pratico**: una volta compreso il flusso, puoi passare agli script specifici (`ottimizzatore_lambda.py`, `risolutore_dwave.py`, ecc.) per esperimenti più strutturati.

## Descrizione dell'Algoritmo

### Algoritmo 1: Espansione del Cluster

L'algoritmo di espansione del cluster affronta il problema di aggiungere esattamente T nuovi punti a un seed di cluster esistente, minimizzando le distanze intra-cluster mentre rispetta i vincoli di cardinalità.

#### Formulazione QUBO

Il problema è formulato come Ottimizzazione Binaria Quadratica Non Vincolata (QUBO):

```text
min Σ(i,j) d_ij * x_i * x_j + λ₂ * (Σx_i - T)²
```

Dove:

* `d_ij`: Distanza tra i punti i e j
* `x_i`: Variabile binaria (1 se il punto i è selezionato, 0 altrimenti)
* `T`: Numero target di punti da selezionare
* `λ₂`: Parametro di penalità per il vincolo di cardinalità

### Ottimizzazione del parametro λ

Il parametro di penalità lambda controlla il peso del vincolo di cardinalità rispetto al termine di distanza.
Valori troppo bassi producono soluzioni non fattibili (numero di punti selezionati diverso da T),
valori troppo alti forzano il vincolo ma rendono il problema numericamente rigido e appiattiscono lo spazio di ricerca.

Nel progetto originale lambda è stato calibrato off-line, istanza per istanza, tramite lo script `ottimizzatore_lambda.py`, che implementa un algoritmo adattivo ispirato ai subgradienti. A ogni iterazione l’algoritmo:

* risolve il QUBO con un risolutore basato su Simulated Annealing per il valore corrente di lambda;
* valuta la soluzione in termini di fattibilità (rispetto del vincolo di cardinalità) e valore della funzione obiettivo;
* aumenta lambda quando la soluzione è infeasible;
* riduce gradualmente il passo di aggiornamento e aggiusta lambda quando trova nuove soluzioni fattibili e migliori.

Durante l’esecuzione viene tenuta traccia dei valori di lambda che producono la stessa soluzione ottima e il valore finale di lambda viene calcolato come media di questi valori. I lambda utilizzati negli esperimenti e riportati nel paper sono salvati nel file `lambda.csv` e, se si utilizzano le stesse istanze, possono essere riusati direttamente senza dover rilanciare l’ottimizzazione.

## Dataset originale

Il caso d’uso bancario si basa sul dataset pubblico **“New Marketing Campaign”**, disponibile su Kaggle:
[https://www.kaggle.com/datasets/mikejimenez24/new-marketing-campaign](https://www.kaggle.com/datasets/mikejimenez24/new-marketing-campaign).

Il dataset contiene oltre 300.000 clienti e descrive, per ciascuno, tre gruppi principali di informazioni:

* **variabili economiche** (es. reddito annuale, importi spesi in diverse categorie di prodotto come vino, carne, pesce, dolci, beni di lusso);
* **variabili di comportamento d’acquisto** (es. numero di acquisti online, su catalogo, in negozio, visite al sito, utilizzo di sconti);
* **variabili demografiche** (es. età, livello di istruzione, stato civile, composizione del nucleo familiare).

Nel progetto QUACK, il dataset è stato preprocessato rimuovendo le variabili non direttamente legate al comportamento di spesa (es. informazioni socio-demografiche, indicatori di campagne marketing) e mantenendo solo le feature che descrivono **quanto** e **come** i clienti acquistano. Le variabili quantitative sono state standardizzate (media zero, varianza unitaria) per rendere coerente il calcolo delle distanze.

## Costruzione delle istanze

A partire dal dataset preprocessato è stata definita una pipeline in due fasi: **segmentazione iniziale** e **generazione delle istanze**.

1. **Clustering e selezione dei clienti rappresentativi**

   * Si applica un algoritmo di clustering (K-Means) sul dataset standardizzato, scegliendo **2 cluster** come numero ottimale.
   * Per ciascun cluster si selezionano i punti più rappresentativi, ossia quelli più vicini al centroide nello spazio delle feature.
   * Da questi si costruiscono tre “pool” di riferimento: **Best 400**, **Best 1000** e **Best 2000** punti, che fungono da base per tutte le istanze successive.

2. **Definizione delle istanze di test**
   A partire dai pool, vengono generate istanze sperimentali che rappresentano diversi scenari di difficoltà. Ogni istanza è caratterizzata da:

   * un **cluster iniziale** I_0, composto da un sottoinsieme di clienti “compatibili” tra loro (seed su cui l’algoritmo deve lavorare);
   * un insieme di **punti candidati all’espansione** C, estratti dal pool e potenzialmente aggiungibili a I_0;
   * una **combinazione di parametri strutturali**, che controllano:

     * la dimensione relativa di I_0 rispetto ai punti totali (ad esempio, configurazioni con I_0 pari al 25%, 50% o 75% del totale);
     * la **percentuale di punti compatibili** dentro C (es. 20%, 40%, 50%, 80%), che introduce un diverso livello di “rumore” nel problema;
     * il **numero complessivo di punti** nell’istanza (es. configurazioni con 4, 8, 16, 32 punti), scelto anche in funzione dei vincoli del quantum annealer.

3. **Struttura logica di una istanza**
   In termini concettuali, ogni istanza contiene almeno:

   * l’elenco dei punti coinvolti (clienti selezionati dal pool) e i relativi **indici**;
   * l’indicazione di quali punti appartengono al **cluster seed** I_0 e quali sono in C;
   * il valore target T, cioè quanti nuovi punti il modello deve aggiungere a I_0;
   * la **matrice di distanza** d_{ij} tra tutti i punti dell’istanza (estratta dalla matrice globale del pool);
   * alcuni **metadati strutturali** (dimensioni del pool, configurazione di rumore, scenario di difficoltà).

   In questa cartella le istanze sono fornite in formato testuale (`.txt`), ma mantengono la stessa struttura logica (seed I_0, candidati C, parametri, matrice di distanza) in una forma più facilmente consultabile e riutilizzabile.

## Avvio Rapido

### Prerequisiti

1. **Python 3.8+** installato
2. **D-Wave Ocean SDK** account e token API (per esecuzione quantistica)
3. **Licenza Gurobi** (per benchmark classico)

### Installazione (indicativa)

L’installazione delle dipendenze può essere gestita a livello di repository principale (`QUACK/`), ad esempio tramite un file `requirements.txt` condiviso. In generale:

```bash
# Clona il repository principale
git clone https://github.com/Bonno9893/QUACK.git
cd QUACK

# Crea ambiente virtuale
python -m venv venv
# Su Linux/macOS:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

Una volta installate le dipendenze, puoi spostarti nella cartella dell’Algoritmo 1 ed eseguire la [demo end-to-end](#demo-end-to-end-uso-rapido) o i singoli script.

## Metriche di performance

Nel paper *“Lookalike Clustering for Customer Segmentation: a Comparative Study of Quantum Annealing and Classical Algorithms”* le prestazioni dei diversi risolutori (QA, SA, Gurobi) sono valutate principalmente rispetto a tre famiglie di metriche.

### 1. Qualità della soluzione

* **Valore della funzione obiettivo**
  Per ogni istanza si considera il valore della funzione obiettivo del problema ridotto (somma di termini lineari e quadratici), indicato come `Obj` per la soluzione ottima trovata da Gurobi e confrontato con i valori ottenuti da QA e SA.

* **Optimality gap (solo per QA)**
  Per le soluzioni **fattibili** di QA viene calcolato l’**optimality gap** rispetto a Gurobi, come scostamento percentuale del valore obiettivo di QA rispetto all’ottimo:

  ```text
  gap(%) = (Obj_QA - Obj_Gurobi) / Obj_Gurobi * 100
  ```

  dove `Obj_QA` è il valore della funzione obiettivo per la migliore soluzione trovata da QA e `Obj_Gurobi` è il valore ottimo trovato da Gurobi (calcolato solo quando QA restituisce almeno una soluzione fattibile per l’istanza).

### 2. Fattibilità e robustezza (QA)

Per il quantum annealer di D-Wave vengono misurate, su più run per ogni istanza, le seguenti quantità:

* **%Feas**: percentuale di esecuzioni in cui QA produce almeno una soluzione fattibile (rispetto al vincolo di cardinalità).
* **%Opt**: percentuale di esecuzioni in cui QA trova una soluzione con valore obiettivo uguale a quello di Gurobi (entro una tolleranza numerica).

Questi indici vengono analizzati per classi di istanze con lo stesso numero di punti (|I'|) e di punti da aggiungere (T'), per valutare come la qualità delle soluzioni di QA degrada all’aumentare della dimensione del problema.

### 3. Efficienza computazionale

Per tutti i risolutori si considera il **tempo di calcolo**:

* **Tempo QA**: tempo di accesso al QPU (somma di programming time e sampling time) per eseguire le chiamate al quantum annealer.
* **Tempo SA**: tempo totale CPU della Simulated Annealing (pre-processing, sampling, post-processing).
* **Tempo Gurobi**: tempo CPU necessario a trovare e certificare l’ottimo del modello ridotto.

Per l’**algoritmo adattivo di ottimizzazione di lambda** vengono inoltre monitorati, per gruppi di istanze con stessa dimensione (|I'|):

* il valore finale di lambda restituito dall’algoritmo;
* il numero medio di iterazioni;
* il tempo medio di esecuzione, confrontato con un metodo a passo fisso.

Nel codice di questa cartella, le informazioni necessarie a ricostruire queste metriche (valori obiettivo, flag di fattibilità, tempi di esecuzione, ecc.) vengono salvate dagli script dei solver e possono essere aggregate per ottenere indicatori analoghi a quelli riportati nel paper.

## Team di Sviluppo

Benedetta Ferrari, Mirko Mucciarini, Filippo Bonafè

## Informazioni Aggiuntive

[https://www.linkedin.com/company/quack-project](https://www.linkedin.com/company/quack-project)

[https://www.re-lab.it/projects/quack](https://www.re-lab.it/projects/quack)

## Pubblicazioni

Ferrari, Benedetta, et al. "Lookalike Clustering for Customer
Segmentation: a Comparative Study of Quantum Annealing and
Classical Algorithms." Proceedings of the Genetic and
Evolutionary Computation Conference Companion. 2025.

---

*Parte del Progetto QUACK (QUAntum Clustering for Knowledge) - Avanzamento delle applicazioni di calcolo quantistico in scenari di clustering del mondo reale*

```

---

Passo successivo, se vuoi: adattiamo **in modo chirurgico** la parte sulla demo ai nomi reali delle funzioni/parametri dentro `demo_quack_banking.py` (ad es. come selezioni l’istanza, come scegli il solver), così il README è perfettamente aderente al codice.
```

