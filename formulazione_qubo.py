"""
Modulo di Formulazione QUBO per Problemi di Clustering

Questo modulo fornisce la funzionalità principale per formulare problemi di clustering
come modelli di Ottimizzazione Binaria Quadratica Non Vincolata (QUBO), specificamente
progettati per il quantum annealing su sistemi D-Wave.

Il focus principale è sull'Algoritmo 1 (Espansione del Cluster), che seleziona esattamente T
punti da aggiungere a un seed di cluster esistente minimizzando le distanze intra-cluster.

Formulazione Matematica:
    min Σ(i,j) d_ij * x_i * x_j + λ₂ * (Σx_i - T)²
    
Dove:
    - d_ij: Distanza tra i punti i e j
    - x_i: Variabile decisionale binaria (1 se il punto i è selezionato, 0 altrimenti)
    - T: Numero target di punti da selezionare (vincolo di cardinalità)
    - λ₂: Parametro di penalità per violazione del vincolo

Autore: Team Progetto QUACK
Data: 2024
Licenza: MIT
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from dimod import BinaryQuadraticModel, QuadraticModel
import logging

# Configurazione del logging
logger = logging.getLogger(__name__)


class FormulatoreQUBO:
    """
    Una classe per formulare problemi di clustering come modelli QUBO.
    
    Questo formulatore crea Modelli Quadratici Binari (BQM) adatti per
    il quantum annealing, con attenzione speciale ai vincoli di cardinalità
    che sono comuni nelle applicazioni di clustering.
    """
    
    def __init__(self, matrice_distanze: np.ndarray):
        """
        Inizializza il formulatore QUBO con una matrice delle distanze.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice quadrata simmetrica delle distanze
                                          a coppie tra tutti i punti nel dataset.
        
        Solleva:
            ValueError: Se la matrice delle distanze non è quadrata o simmetrica.
        """
        self._valida_matrice_distanze(matrice_distanze)
        self.matrice_distanze = matrice_distanze
        self.n_punti = matrice_distanze.shape[0]
        
        logger.info(f"Formulatore QUBO inizializzato per {self.n_punti} punti")
    
    def _valida_matrice_distanze(self, matrice: np.ndarray) -> None:
        """
        Valida che la matrice delle distanze sia formattata correttamente.
        
        Parametri:
            matrice (np.ndarray): Matrice da validare
            
        Solleva:
            ValueError: Se la matrice non è quadrata o non simmetrica
        """
        # Controlla che sia bidimensionale
        if matrice.ndim != 2:
            raise ValueError("La matrice delle distanze deve essere bidimensionale")
        
        # Controlla che sia quadrata
        if matrice.shape[0] != matrice.shape[1]:
            raise ValueError("La matrice delle distanze deve essere quadrata")
        
        # Controlla la simmetria (con piccola tolleranza per errori numerici)
        if not np.allclose(matrice, matrice.T, rtol=1e-10):
            raise ValueError("La matrice delle distanze deve essere simmetrica")
        
        # Controlla distanze non negative
        if np.any(matrice < 0):
            logger.warning("La matrice delle distanze contiene valori negativi")
    
    def crea_qubo_espansione_cluster(
        self,
        cluster_seed: np.ndarray,
        dimensione_target: int,
        penalita_lambda: float,
        punti_candidati: Optional[np.ndarray] = None
    ) -> BinaryQuadraticModel:
        """
        Crea un modello QUBO per il problema di espansione del cluster.
        
        Questa formulazione mira a selezionare esattamente 'dimensione_target' punti dal
        set di candidati che sono più vicini al cluster seed, minimizzando la
        distanza intra-cluster totale.
        
        Parametri:
            cluster_seed (np.ndarray): Indici dei punti già nel cluster
            dimensione_target (int): Numero di punti da aggiungere al cluster
            penalita_lambda (float): Peso della penalità per il vincolo di cardinalità
            punti_candidati (np.ndarray, opzionale): Indici dei punti candidati.
                                                     Se None, tutti i punti non-seed
                                                     sono candidati.
        
        Ritorna:
            BinaryQuadraticModel: Il modello QUBO pronto per essere risolto
        
        Esempio:
            >>> formulatore = FormulatoreQUBO(matrice_distanze)
            >>> seed = np.array([0, 1, 2])  # Cluster iniziale
            >>> bqm = formulatore.crea_qubo_espansione_cluster(
            ...     cluster_seed=seed,
            ...     dimensione_target=5,  # Aggiungi 5 punti
            ...     penalita_lambda=2.0
            ... )
        """
        # Determina i punti candidati (tutti i punti non nel cluster seed)
        if punti_candidati is None:
            tutti_punti = set(range(self.n_punti))
            set_seed = set(cluster_seed)
            punti_candidati = np.array(list(tutti_punti - set_seed))
        
        n_candidati = len(punti_candidati)
        
        logger.info(f"Creazione QUBO per selezionare {dimensione_target} da "
                   f"{n_candidati} candidati per espandere cluster di dimensione {len(cluster_seed)}")
        
        # Inizializza il Modello Quadratico Binario
        bqm = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
        
        # === 1. Termini di minimizzazione delle distanze ===
        # Aggiungi termini quadratici per le distanze tra punti selezionati
        for i in range(n_candidati):
            for j in range(i + 1, n_candidati):
                punto_i = punti_candidati[i]
                punto_j = punti_candidati[j]
                distanza = self.matrice_distanze[punto_i, punto_j]
                
                # Aggiungi termine di interazione: selezionare sia i che j comporta la loro distanza
                bqm.add_quadratic(f"x_{i}", f"x_{j}", distanza)
        
        # Aggiungi termini lineari per le distanze al cluster seed
        for i in range(n_candidati):
            punto_i = punti_candidati[i]
            distanza_al_seed = sum(
                self.matrice_distanze[punto_i, punto_seed]
                for punto_seed in cluster_seed
            )
            bqm.add_linear(f"x_{i}", distanza_al_seed)
        
        # === 2. Applicazione del vincolo di cardinalità ===
        # Termine di penalità: λ₂ * (Σx_i - T)²
        # Espanso: λ₂ * (ΣΣ x_i*x_j - 2T*Σx_i + T²)
        
        # Termini di penalità quadratici
        for i in range(n_candidati):
            for j in range(n_candidati):
                if i != j:
                    # Interazione tra variabili diverse
                    bqm.add_quadratic(f"x_{i}", f"x_{j}", penalita_lambda)
                else:
                    # Auto-interazione (x_i * x_i = x_i per variabili binarie)
                    bqm.add_linear(f"x_{i}", penalita_lambda)
        
        # Termini di penalità lineari
        for i in range(n_candidati):
            bqm.add_linear(f"x_{i}", -2 * penalita_lambda * dimensione_target)
        
        # Termine costante (non influenza l'ottimizzazione ma necessario per il calcolo dell'energia)
        bqm.offset += penalita_lambda * dimensione_target * dimensione_target
        
        logger.info(f"Modello QUBO creato con {len(bqm.variables)} variabili e "
                   f"{len(bqm.quadratic)} termini quadratici")
        
        return bqm
    
    def crea_qubo_clustering_globale(
        self,
        n_cluster: int,
        penalita_lambda: float
    ) -> BinaryQuadraticModel:
        """
        Crea un modello QUBO per il clustering globale (Algoritmo 2).
        
        Questa formulazione assegna tutti i punti a uno di k cluster, assicurando
        che ogni punto appartenga a esattamente un cluster mentre minimizza le
        distanze intra-cluster totali.
        
        Parametri:
            n_cluster (int): Numero di cluster da formare
            penalita_lambda (float): Peso della penalità per i vincoli di assegnazione
        
        Ritorna:
            BinaryQuadraticModel: Il modello QUBO per il clustering globale
        
        Nota:
            Questo usa variabili binarie x_{i,k} dove x_{i,k} = 1 significa
            che il punto i è assegnato al cluster k.
        """
        bqm = BinaryQuadraticModel({}, {}, 0.0, "BINARY")
        
        logger.info(f"Creazione QUBO per clustering globale di {self.n_punti} punti "
                   f"in {n_cluster} cluster")
        
        # === 1. Minimizzazione delle distanze all'interno dei cluster ===
        for k in range(n_cluster):
            for i in range(self.n_punti):
                for j in range(i + 1, self.n_punti):
                    # Costo di avere entrambi i punti i e j nel cluster k
                    var_i = f"x_{i}_{k}"
                    var_j = f"x_{j}_{k}"
                    bqm.add_quadratic(var_i, var_j, self.matrice_distanze[i, j])
        
        # === 2. Vincolo di assegnazione unica ===
        # Ogni punto deve essere assegnato a esattamente un cluster
        # Penalità: λ * (Σ_k x_{i,k} - 1)² per ogni punto i
        
        for i in range(self.n_punti):
            vars_i = [f"x_{i}_{k}" for k in range(n_cluster)]
            
            # Aggiungi termini di penalità quadratici per violazione del vincolo
            for k1 in range(n_cluster):
                for k2 in range(k1 + 1, n_cluster):
                    # Penalizza l'assegnazione del punto i a più cluster
                    bqm.add_quadratic(vars_i[k1], vars_i[k2], 2 * penalita_lambda)
            
            # Aggiungi termini lineari
            for var in vars_i:
                bqm.add_linear(var, -2 * penalita_lambda)
            
            # Aggiungi termine costante
            bqm.offset += penalita_lambda
        
        logger.info(f"QUBO per clustering globale creato con {len(bqm.variables)} variabili")
        
        return bqm
    
    def calcola_breakdown_energia(
        self,
        soluzione: Dict[str, int],
        bqm: BinaryQuadraticModel
    ) -> Dict[str, float]:
        """
        Calcola il breakdown dell'energia per una data soluzione.
        
        Questo aiuta a capire quanto ogni componente (distanza vs. penalità)
        contribuisce all'energia totale.
        
        Parametri:
            soluzione (Dict[str, int]): Dizionario della soluzione binaria
            bqm (BinaryQuadraticModel): Il modello QUBO
        
        Ritorna:
            Dict[str, float]: Breakdown dell'energia con chiavi:
                - 'totale': Energia totale
                - 'distanza': Energia dai termini di distanza
                - 'penalita': Energia dalle penalità dei vincoli
        """
        energia_totale = bqm.energy(soluzione)
        
        # Calcola contributo delle distanze
        # (Questo è approssimativo - assume di poter identificare i termini di penalità dalla magnitudine)
        energia_distanza = 0.0
        energia_penalita = 0.0
        
        # Termini lineari
        for var, coeff in bqm.linear.items():
            if var in soluzione:
                contributo = coeff * soluzione[var]
                # Euristica: coefficienti negativi grandi sono probabilmente penalità
                if coeff < -1.0:
                    energia_penalita += contributo
                else:
                    energia_distanza += contributo
        
        # Termini quadratici
        for (var1, var2), coeff in bqm.quadratic.items():
            if var1 in soluzione and var2 in soluzione:
                contributo = coeff * soluzione[var1] * soluzione[var2]
                # Euristica: coefficienti positivi grandi sono probabilmente penalità
                if coeff > 1.0:
                    energia_penalita += contributo
                else:
                    energia_distanza += contributo
        
        return {
            'totale': energia_totale,
            'distanza': energia_distanza,
            'penalita': energia_penalita,
            'offset': bqm.offset
        }
    
    def valida_soluzione(
        self,
        soluzione: Dict[str, int],
        dimensione_target: int
    ) -> Tuple[bool, int]:
        """
        Valida se una soluzione soddisfa il vincolo di cardinalità.
        
        Parametri:
            soluzione (Dict[str, int]): Dizionario della soluzione binaria
            dimensione_target (int): Numero atteso di punti selezionati
        
        Ritorna:
            Tuple[bool, int]: (è_valida, dimensione_effettiva)
        """
        conteggio_selezionati = sum(soluzione.values())
        è_valida = (conteggio_selezionati == dimensione_target)
        
        if not è_valida:
            logger.warning(f"La soluzione seleziona {conteggio_selezionati} punti, "
                          f"attesi {dimensione_target}")
        
        return è_valida, conteggio_selezionati


def crea_qubo_consapevole_distanze(
    matrice_distanze: np.ndarray,
    n_selezionare: int,
    lambda_inizio: float = 1.0,
    lambda_fine: float = 10.0,
    adattivo: bool = True
) -> Tuple[BinaryQuadraticModel, float]:
    """
    Crea un QUBO con selezione adattiva del parametro lambda.
    
    Questa funzione determina automaticamente un buon valore di lambda basato
    sulla scala delle distanze nel problema.
    
    Parametri:
        matrice_distanze (np.ndarray): Matrice delle distanze a coppie
        n_selezionare (int): Numero di punti da selezionare
        lambda_inizio (float): Valore lambda iniziale per la ricerca
        lambda_fine (float): Valore lambda massimo da considerare
        adattivo (bool): Se usare la selezione adattiva di lambda
    
    Ritorna:
        Tuple[BinaryQuadraticModel, float]: (modello QUBO, valore lambda selezionato)
    """
    formulatore = FormulatoreQUBO(matrice_distanze)
    
    if adattivo:
        # Stima un buon lambda basato sulla scala delle distanze
        distanza_media = np.mean(matrice_distanze[matrice_distanze > 0])
        deviazione_std = np.std(matrice_distanze[matrice_distanze > 0])
        
        # Lambda dovrebbe essere comparabile alla scala delle distanze
        lambda_stimato = distanza_media + deviazione_std
        
        logger.info(f"Selezione lambda adattiva: λ stimato = {lambda_stimato:.3f}")
        
        # Crea QUBO con lambda stimato
        seed = np.array([])  # Seed vuoto per problema di pura selezione
        bqm = formulatore.crea_qubo_espansione_cluster(
            cluster_seed=seed,
            dimensione_target=n_selezionare,
            penalita_lambda=lambda_stimato
        )
        
        return bqm, lambda_stimato
    else:
        # Usa il valore lambda fornito (metà del range)
        valore_lambda = (lambda_inizio + lambda_fine) / 2
        seed = np.array([])
        
        bqm = formulatore.crea_qubo_espansione_cluster(
            cluster_seed=seed,
            dimensione_target=n_selezionare,
            penalita_lambda=valore_lambda
        )
        
        return bqm, valore_lambda
