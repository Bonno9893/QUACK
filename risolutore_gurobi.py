"""
Modulo Risolutore Gurobi

Questo modulo fornisce un risolutore esatto usando Gurobi per problemi di clustering.
Serve come benchmark per confrontare le soluzioni quantistiche con l'ottimo classico.

Basato sui file gurobi_model.py e Performance_Gurobi.py del progetto QUACK.

Autore: Team Progetto QUACK
Data: 2024
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_DISPONIBILE = True
except ImportError:
    GUROBI_DISPONIBILE = False
    logger.warning("Gurobi non installato. Il risolutore Gurobi non sarà disponibile.")

from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


@dataclass 
class ConfigurazioneGurobi:
    """Configurazione per il risolutore Gurobi."""
    time_limit: int = 60  # Limite tempo in secondi
    mip_gap: float = 0.01  # Gap ottimalità MIP
    threads: int = 0  # 0 per auto
    output_flag: int = 0  # 0 per silenzioso, 1 per verboso
    
    
class RisolutoreGurobi:
    """
    Risolutore esatto per problemi di clustering usando Gurobi.
    
    Questo risolutore fornisce soluzioni ottime (o quasi-ottime) per
    problemi di clustering, servendo come benchmark per i metodi quantistici.
    """
    
    def __init__(self, config: Optional[ConfigurazioneGurobi] = None):
        """
        Inizializza il risolutore Gurobi.
        
        Parametri:
            config (ConfigurazioneGurobi, opzionale): Configurazione del risolutore
            
        Solleva:
            ImportError: Se Gurobi non è installato
        """
        if not GUROBI_DISPONIBILE:
            raise ImportError("Gurobi non disponibile. Installare gurobipy.")
        
        self.config = config or ConfigurazioneGurobi()
        logger.info(f"Risolutore Gurobi inizializzato con time limit {self.config.time_limit}s")
    
    def risolvi_clustering(
        self,
        matrice_distanze: np.ndarray,
        n_cluster: int,
        assegnazione_iniziale: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Risolve problema di clustering globale con Gurobi.
        
        Formula e risolve il problema di clustering come programma
        intero misto (MIP) per trovare l'assegnazione ottimale.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice distanze a coppie
            n_cluster (int): Numero di cluster da formare
            assegnazione_iniziale (np.ndarray, opzionale): Soluzione iniziale warm start
        
        Ritorna:
            Dict con dettagli soluzione:
                - 'clustering': Assegnazione cluster per ogni punto
                - 'obiettivo': Valore funzione obiettivo
                - 'tempo_risoluzione': Tempo di risoluzione
                - 'gap': Gap di ottimalità
                - 'status': Status del solver
        """
        n_punti = matrice_distanze.shape[0]
        
        # Normalizza matrice distanze se necessario
        matrice_norm = self._normalizza_matrice_distanze(matrice_distanze)
        
        logger.info(f"Risoluzione clustering: {n_punti} punti in {n_cluster} cluster")
        
        # Crea modello Gurobi
        modello = gp.Model("clustering")
        
        # Imposta parametri
        modello.setParam("TimeLimit", self.config.time_limit)
        modello.setParam("MIPGap", self.config.mip_gap)
        modello.setParam("OutputFlag", self.config.output_flag)
        if self.config.threads > 0:
            modello.setParam("Threads", self.config.threads)
        
        # === Variabili decisionali ===
        # x[i,k] = 1 se punto i è assegnato a cluster k
        x = modello.addVars(n_punti, n_cluster, vtype=GRB.BINARY, name="x")
        
        # y[i,j,k] = 1 se entrambi i punti i e j sono nel cluster k
        # Solo per i < j per evitare duplicazioni
        y = modello.addVars(
            [(i, j, k) for i in range(n_punti) 
             for j in range(i+1, n_punti) 
             for k in range(n_cluster)],
            vtype=GRB.BINARY,
            name="y"
        )
        
        # === Vincoli ===
        
        # Ogni punto deve essere assegnato a esattamente un cluster
        for i in range(n_punti):
            modello.addConstr(
                gp.quicksum(x[i, k] for k in range(n_cluster)) == 1,
                f"assegnazione_{i}"
            )
        
        # Linearizzazione: y[i,j,k] = x[i,k] * x[j,k]
        for i in range(n_punti):
            for j in range(i+1, n_punti):
                for k in range(n_cluster):
                    # y[i,j,k] <= x[i,k]
                    modello.addConstr(y[i,j,k] <= x[i,k], f"lin1_{i}_{j}_{k}")
                    # y[i,j,k] <= x[j,k]
                    modello.addConstr(y[i,j,k] <= x[j,k], f"lin2_{i}_{j}_{k}")
                    # y[i,j,k] >= x[i,k] + x[j,k] - 1
                    modello.addConstr(
                        y[i,j,k] >= x[i,k] + x[j,k] - 1,
                        f"lin3_{i}_{j}_{k}"
                    )
        
        # Vincolo opzionale: dimensione minima cluster (evita cluster vuoti)
        min_punti_per_cluster = max(1, n_punti // (n_cluster * 2))
        for k in range(n_cluster):
            modello.addConstr(
                gp.quicksum(x[i, k] for i in range(n_punti)) >= min_punti_per_cluster,
                f"min_dimensione_{k}"
            )
        
        # === Funzione obiettivo ===
        # Minimizza somma distanze intra-cluster
        obiettivo = gp.quicksum(
            matrice_norm[i, j] * y[i,j,k]
            for i in range(n_punti)
            for j in range(i+1, n_punti)
            for k in range(n_cluster)
        )
        
        modello.setObjective(obiettivo, GRB.MINIMIZE)
        
        # Warm start se fornito
        if assegnazione_iniziale is not None:
            for i in range(n_punti):
                for k in range(n_cluster):
                    x[i, k].Start = 1 if assegnazione_iniziale[i] == k else 0
        
        # === Risoluzione ===
        tempo_inizio = time.time()
        modello.optimize()
        tempo_risoluzione = time.time() - tempo_inizio
        
        # === Estrazione risultati ===
        if modello.SolCount == 0:
            logger.warning("Nessuna soluzione trovata da Gurobi")
            return {
                'clustering': None,
                'obiettivo': float('inf'),
                'tempo_risoluzione': tempo_risoluzione,
                'gap': float('inf'),
                'status': modello.Status,
                'fattibile': False
            }
        
        # Estrai assegnazione cluster
        clustering = np.zeros(n_punti, dtype=int)
        for i in range(n_punti):
            for k in range(n_cluster):
                if x[i, k].X > 0.5:  # Valore binario
                    clustering[i] = k
                    break
        
        # Calcola valore obiettivo con matrice originale
        valore_obiettivo_originale = self._calcola_costo_clustering(
            matrice_distanze, clustering
        )
        
        risultati = {
            'clustering': clustering,
            'obiettivo': valore_obiettivo_originale,
            'obiettivo_normalizzato': modello.ObjVal,
            'tempo_risoluzione': tempo_risoluzione,
            'gap': modello.MIPGap,
            'status': modello.Status,
            'fattibile': True,
            'dimensioni_cluster': np.bincount(clustering)
        }
        
        logger.info(f"Gurobi risolto: obiettivo={valore_obiettivo_originale:.3f}, "
                   f"tempo={tempo_risoluzione:.2f}s, gap={modello.MIPGap:.4f}")
        
        return risultati
    
    def risolvi_espansione_cluster(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int
    ) -> Dict[str, Any]:
        """
        Risolve problema di espansione cluster con Gurobi.
        
        Formula il problema di selezionare esattamente n_espandere punti
        da aggiungere al cluster seed minimizzando la distanza totale.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice distanze
            cluster_seed (np.ndarray): Indici punti nel cluster iniziale
            n_espandere (int): Numero punti da aggiungere
        
        Ritorna:
            Dict con risultati
        """
        n_punti = matrice_distanze.shape[0]
        
        # Punti candidati (tutti tranne seed)
        candidati = [i for i in range(n_punti) if i not in cluster_seed]
        n_candidati = len(candidati)
        
        logger.info(f"Espansione cluster: seed={len(cluster_seed)}, "
                   f"espandi={n_espandere}, candidati={n_candidati}")
        
        # Crea modello
        modello = gp.Model("espansione_cluster")
        modello.setParam("TimeLimit", self.config.time_limit)
        modello.setParam("OutputFlag", self.config.output_flag)
        
        # Variabili: x[i] = 1 se candidato i è selezionato
        x = modello.addVars(n_candidati, vtype=GRB.BINARY, name="x")
        
        # Vincolo: seleziona esattamente n_espandere punti
        modello.addConstr(
            gp.quicksum(x[i] for i in range(n_candidati)) == n_espandere,
            "cardinalita"
        )
        
        # Funzione obiettivo: minimizza distanza totale
        obiettivo = gp.QuadExpr()
        
        # Distanze tra punti selezionati
        for i in range(n_candidati):
            for j in range(i+1, n_candidati):
                idx_i = candidati[i]
                idx_j = candidati[j]
                obiettivo += matrice_distanze[idx_i, idx_j] * x[i] * x[j]
        
        # Distanze dai candidati al seed
        for i in range(n_candidati):
            idx_candidato = candidati[i]
            distanza_al_seed = sum(
                matrice_distanze[idx_candidato, idx_seed]
                for idx_seed in cluster_seed
            )
            obiettivo += distanza_al_seed * x[i]
        
        modello.setObjective(obiettivo, GRB.MINIMIZE)
        
        # Risolvi
        tempo_inizio = time.time()
        modello.optimize()
        tempo_risoluzione = time.time() - tempo_inizio
        
        # Estrai risultati
        if modello.SolCount == 0:
            return {
                'indici_selezionati': np.array([]),
                'obiettivo': float('inf'),
                'tempo_risoluzione': tempo_risoluzione,
                'fattibile': False
            }
        
        # Estrai punti selezionati
        indici_selezionati = []
        for i in range(n_candidati):
            if x[i].X > 0.5:
                indici_selezionati.append(candidati[i])
        
        indici_selezionati = np.array(indici_selezionati)
        
        return {
            'indici_selezionati': indici_selezionati,
            'obiettivo': modello.ObjVal,
            'tempo_risoluzione': tempo_risoluzione,
            'gap': modello.MIPGap,
            'fattibile': len(indici_selezionati) == n_espandere
        }
    
    def benchmark_performance(
        self,
        istanze: List[Dict],
        percorso_output: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Esegue benchmark su multiple istanze.
        
        Basato su Performance_Gurobi.py del progetto.
        
        Parametri:
            istanze: Lista istanze da risolvere
            percorso_output: Percorso per salvare risultati Excel
        
        Ritorna:
            DataFrame con metriche performance
        """
        righe = []
        
        for istanza in istanze:
            nome_istanza = istanza.get('nome', f"istanza_{istanza.get('id', 0)}")
            logger.info(f"Processando {nome_istanza}")
            
            matrice_distanze = istanza['matrice_distanze']
            n_punti = matrice_distanze.shape[0]
            n_cluster = istanza.get('n_cluster', 3)
            y_true = istanza.get('y_true')
            
            # Risolvi con Gurobi
            risultati = self.risolvi_clustering(
                matrice_distanze=matrice_distanze,
                n_cluster=n_cluster
            )
            
            if risultati['clustering'] is not None:
                # Calcola metriche
                riga = {
                    'nome_file': nome_istanza,
                    'n_punti': n_punti,
                    'n_cluster': n_cluster,
                    'gurobi_obiettivo': risultati['obiettivo'],
                    'gurobi_gap': risultati['gap'],
                    'gurobi_tempo_s': risultati['tempo_risoluzione'],
                    'gurobi_fattibile': risultati['fattibile']
                }
                
                # Calcola ARI se ground truth disponibile
                if y_true is not None:
                    ari = adjusted_rand_score(y_true, risultati['clustering'])
                    riga['gurobi_ari'] = ari
                
                # Aggiungi assegnazione come stringa
                riga['gurobi_assegnazione'] = ','.join(
                    map(str, risultati['clustering'].tolist())
                )
                
                righe.append(riga)
            else:
                logger.warning(f"Nessuna soluzione per {nome_istanza}")
        
        # Crea DataFrame
        df = pd.DataFrame(righe)
        
        # Salva se richiesto
        if percorso_output:
            df.to_excel(percorso_output, index=False)
            logger.info(f"Risultati salvati in {percorso_output}")
        
        return df
    
    def _normalizza_matrice_distanze(self, matrice: np.ndarray) -> np.ndarray:
        """
        Normalizza matrice distanze.
        
        Parametri:
            matrice: Matrice da normalizzare
        
        Ritorna:
            Matrice normalizzata
        """
        max_val = matrice.max()
        if max_val > 0:
            return matrice / max_val
        return matrice.copy()
    
    def _calcola_costo_clustering(
        self,
        matrice_distanze: np.ndarray,
        clustering: np.ndarray
    ) -> float:
        """
        Calcola costo totale del clustering (somma distanze intra-cluster).
        
        Parametri:
            matrice_distanze: Matrice distanze
            clustering: Assegnazione cluster
        
        Ritorna:
            Costo totale
        """
        costo = 0.0
        for k in np.unique(clustering):
            idx = np.where(clustering == k)[0]
            if len(idx) > 1:
                # Somma tutte le distanze a coppie nel cluster
                for i in range(len(idx)):
                    for j in range(i+1, len(idx)):
                        costo += matrice_distanze[idx[i], idx[j]]
        
        return costo
