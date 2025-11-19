"""
Modulo Risolutore Simulated Annealing

Questo modulo fornisce un risolutore di simulated annealing classico per problemi
di clustering, servendo sia come risolutore standalone che come strumento per
l'ottimizzazione dei parametri nella pipeline quantistica.

L'implementazione usa il package neal di D-Wave per consistenza con la
formulazione quantistica, permettendo confronto diretto dei risultati.

Basato sui notebook Simulated-tests.ipynb del progetto QUACK.

Autore: Team Progetto QUACK
Data: 2024
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
import dimod

logger = logging.getLogger(__name__)


@dataclass
class ConfigurazioneSA:
    """Configurazione per risolutore Simulated Annealing."""
    num_reads: int = 1000
    num_sweeps: int = 1000
    beta_range: Tuple[float, float] = (0.1, 10.0)
    beta_schedule_type: str = 'geometric'
    seed: Optional[int] = 42
    initial_states_generator: str = 'random'
    chain_strength: float = 10000  # CS dal notebook originale


class RisolutoreSimulatedAnnealing:
    """
    Risolutore Simulated Annealing per problemi di clustering.
    
    Questo risolutore fornisce un baseline classico per confronto con
    risultati del quantum annealing. Usa la stessa formulazione QUBO
    del risolutore quantistico per confronto diretto.
    """
    
    def __init__(self, config: Optional[ConfigurazioneSA] = None):
        """
        Inizializza il risolutore Simulated Annealing.
        
        Parametri:
            config (ConfigurazioneSA, opzionale): Oggetto configurazione. Se None, usa defaults.
        """
        self.config = config or ConfigurazioneSA()
        self.sampler = SimulatedAnnealingSampler()
        
        logger.info(f"Risolutore SA inizializzato con {self.config.num_reads} reads")
    
    def risolvi_espansione_cluster_v3(
        self,
        n_punti: int,
        matrice_distanze: np.ndarray,
        indici_i0: List[int],
        n_punti_da_aggiungere: int,
        lambda2: float,
        usa_dwave: bool = False,
        num_reads: int = 100
    ) -> Tuple[Dict, BinaryQuadraticModel, Any]:
        """
        Risolve problema di espansione cluster (versione 3 dal notebook).
        
        Questa è l'implementazione principale usata nei test, che corrisponde
        alla funzione cluster_points_v3 nel notebook Simulated-tests.ipynb.
        
        Parametri:
            n_punti: Numero totale di punti
            matrice_distanze: Matrice delle distanze
            indici_i0: Indici dei punti nel cluster iniziale
            n_punti_da_aggiungere: Numero di punti da aggiungere
            lambda2: Parametro di penalità lambda
            usa_dwave: Se True usa D-Wave, altrimenti Simulated Annealing
            num_reads: Numero di reads/campioni
        
        Ritorna:
            Tuple: (miglior_campione, bqm, sampler_info)
        """
        # Crea modello BQM
        bqm = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')
        
        # Lista di tutti gli indici disponibili (esclusi quelli già nel cluster)
        indici_disponibili = [i for i in range(n_punti) if i not in indici_i0]
        
        # === Costruzione del QUBO ===
        
        # Termini quadratici per distanze tra punti da aggiungere
        for i, idx_i in enumerate(indici_disponibili):
            for j, idx_j in enumerate(indici_disponibili):
                if i < j:
                    # Distanza tra punti candidati
                    distanza = matrice_distanze[idx_i, idx_j]
                    bqm.add_quadratic(f'x_{idx_i}', f'x_{idx_j}', distanza)
        
        # Termini lineari per distanze dai punti candidati al cluster seed
        for idx_candidato in indici_disponibili:
            distanza_totale_al_seed = 0
            for idx_seed in indici_i0:
                distanza_totale_al_seed += matrice_distanze[idx_candidato, idx_seed]
            
            if distanza_totale_al_seed > 0:
                bqm.add_linear(f'x_{idx_candidato}', distanza_totale_al_seed)
        
        # === Vincolo di cardinalità ===
        # Penalità per selezionare esattamente n_punti_da_aggiungere punti
        
        # Termini quadratici del vincolo
        for idx_i in indici_disponibili:
            for idx_j in indici_disponibili:
                if idx_i != idx_j:
                    bqm.add_quadratic(f'x_{idx_i}', f'x_{idx_j}', lambda2)
                else:
                    # Auto-interazione (x_i * x_i = x_i per variabili binarie)
                    bqm.add_linear(f'x_{idx_i}', lambda2)
        
        # Termini lineari del vincolo
        for idx in indici_disponibili:
            bqm.add_linear(f'x_{idx}', -2 * lambda2 * n_punti_da_aggiungere)
        
        # Termine costante
        bqm.offset += lambda2 * n_punti_da_aggiungere * n_punti_da_aggiungere
        
        # === Risoluzione ===
        if usa_dwave:
            # Importa D-Wave sampler
            from dwave.system import DWaveSampler, EmbeddingComposite
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(bqm, num_reads=num_reads)
        else:
            # Usa Simulated Annealing
            sampleset = self.sampler.sample(
                bqm,
                num_reads=num_reads,
                num_sweeps=self.config.num_sweeps,
                beta_range=self.config.beta_range,
                beta_schedule_type=self.config.beta_schedule_type,
                seed=self.config.seed
            )
        
        # Ottieni miglior soluzione
        miglior_campione = sampleset.first.sample
        
        # Info sul sampler per logging
        sampler_info = sampleset.info if hasattr(sampleset, 'info') else {}
        
        logger.info(f"SA risoluzione: energia={sampleset.first.energy:.3f}, "
                   f"occorrenze={sampleset.first.num_occurrences}")
        
        return miglior_campione, bqm, sampler_info
    
    def controlla_soluzione(
        self,
        campione: Dict[str, int],
        n_totale_atteso: int,
        matrice_distanze: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Controlla se la soluzione è fattibile e calcola funzione obiettivo.
        
        Corrisponde alla funzione check_solution del notebook.
        
        Parametri:
            campione: Soluzione binaria
            n_totale_atteso: Numero totale di punti attesi nel cluster
            matrice_distanze: Matrice distanze
        
        Ritorna:
            Tuple: (è_fattibile, valore_funzione_obiettivo)
        """
        # Estrai punti selezionati
        punti_selezionati = [
            int(var.split('_')[1]) 
            for var, val in campione.items() 
            if val == 1
        ]
        
        # Controlla vincolo di cardinalità
        è_fattibile = (len(punti_selezionati) == n_totale_atteso)
        
        # Calcola funzione obiettivo (somma distanze intra-cluster)
        funzione_obiettivo = 0.0
        for i in range(len(punti_selezionati)):
            for j in range(i + 1, len(punti_selezionati)):
                idx_i = punti_selezionati[i]
                idx_j = punti_selezionati[j]
                funzione_obiettivo += matrice_distanze[idx_i, idx_j]
        
        return è_fattibile, funzione_obiettivo
    
    def esegui_test_batch(
        self,
        istanze: List[Dict],
        lista_lambda: List[float],
        num_iterazioni: int = 10,
        percorso_output: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Esegue test batch su multiple istanze (dal notebook Simulated-tests).
        
        Parametri:
            istanze: Lista di istanze da testare
            lista_lambda: Lista valori lambda da usare
            num_iterazioni: Numero iterazioni per istanza
            percorso_output: Percorso file Excel output
        
        Ritorna:
            DataFrame con risultati
        """
        lista_output = []
        
        for i, istanza in enumerate(istanze):
            logger.info(f"Processando istanza {i}")
            
            # Estrai dati istanza
            matrice_distanze = istanza['matrice_distanze']
            indici_i0 = istanza['cluster_seed']
            n_punti_da_aggiungere = istanza['n_espandere']
            
            # Usa lambda corrispondente
            lambda2 = lista_lambda[i] if i < len(lista_lambda) else lista_lambda[-1]
            
            dati_output = {
                'istanza': f'istanza_{i}',
                'lambda2': lambda2
            }
            
            # Esegui multiple iterazioni
            for iterazione in range(num_iterazioni):
                tempo_inizio = time.time()
                
                # Risolvi con SA
                miglior_campione, bqm, sampler_info = self.risolvi_espansione_cluster_v3(
                    n_punti=matrice_distanze.shape[0],
                    matrice_distanze=matrice_distanze,
                    indici_i0=indici_i0,
                    n_punti_da_aggiungere=n_punti_da_aggiungere,
                    lambda2=lambda2,
                    usa_dwave=False,
                    num_reads=100
                )
                
                tempo_fine = time.time()
                
                # Controlla soluzione
                è_fattibile, valore_of = self.controlla_soluzione(
                    miglior_campione,
                    n_punti_da_aggiungere + len(indici_i0),
                    matrice_distanze
                )
                
                # Registra risultati
                dati_output['iterazione'] = iterazione
                dati_output['fattibile'] = è_fattibile
                dati_output['OF'] = valore_of
                dati_output['tempo_esecuzione'] = tempo_fine - tempo_inizio
                
                # Aggiungi info dal sampler
                if isinstance(sampler_info, dict):
                    for chiave, valore in sampler_info.items():
                        if isinstance(valore, dict):
                            for sub_chiave, sub_valore in valore.items():
                                dati_output[f"{chiave}.{sub_chiave}"] = sub_valore
                        else:
                            dati_output[chiave] = valore
                
                lista_output.append(dati_output.copy())
        
        # Crea DataFrame
        df = pd.DataFrame(lista_output)
        
        # Esporta in Excel se richiesto
        if percorso_output:
            df.to_excel(percorso_output, index=False)
            logger.info(f"Risultati salvati in {percorso_output}")
        
        return df
    
    def risolvi(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        penalita_lambda: float,
        punti_candidati: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Interfaccia standard per risoluzione espansione cluster.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice distanze a coppie
            cluster_seed (np.ndarray): Indici punti cluster seed
            n_espandere (int): Numero punti da aggiungere
            penalita_lambda (float): Parametro penalità
            punti_candidati (np.ndarray, opzionale): Punti candidati per espansione
        
        Ritorna:
            Dict contenente dettagli soluzione
        """
        # Usa implementazione v3
        miglior_campione, bqm, sampler_info = self.risolvi_espansione_cluster_v3(
            n_punti=matrice_distanze.shape[0],
            matrice_distanze=matrice_distanze,
            indici_i0=cluster_seed.tolist(),
            n_punti_da_aggiungere=n_espandere,
            lambda2=penalita_lambda,
            usa_dwave=False,
            num_reads=self.config.num_reads
        )
        
        # Estrai indici selezionati
        indici_selezionati = np.array([
            int(var.split('_')[1])
            for var, val in miglior_campione.items()
            if val == 1
        ])
        
        # Controlla fattibilità
        è_fattibile, valore_of = self.controlla_soluzione(
            miglior_campione,
            n_espandere,
            matrice_distanze
        )
        
        # Calcola statistiche soluzione
        statistiche_soluzione = self._calcola_statistiche_soluzione(
            bqm, n_espandere
        )
        
        return {
            'miglior_campione': miglior_campione,
            'energia': bqm.energy(miglior_campione),
            'indici_selezionati': indici_selezionati,
            'fattibile': è_fattibile,
            'valore_funzione_obiettivo': valore_of,
            'num_occorrenze': sampler_info.get('num_occurrences', 1),
            'statistiche_soluzione': statistiche_soluzione,
            'info_sampler': sampler_info
        }
    
    def _calcola_statistiche_soluzione(
        self,
        bqm: BinaryQuadraticModel,
        dimensione_target: int
    ) -> Dict[str, Any]:
        """
        Calcola statistiche sulla distribuzione delle soluzioni.
        
        Parametri:
            bqm: Modello BQM
            dimensione_target: Numero atteso punti selezionati
        
        Ritorna:
            Dict con statistiche soluzione
        """
        # Esegui campionamento multiplo per statistiche
        sampleset = self.sampler.sample(
            bqm,
            num_reads=100,
            num_sweeps=100,
            seed=42
        )
        
        conteggio_fattibili = 0
        distribuzione_energie = []
        distribuzione_dimensioni = []
        
        for sample in sampleset:
            # Conta variabili selezionate
            selezionati = sum(sample.sample.values())
            distribuzione_dimensioni.append(selezionati)
            
            # Controlla fattibilità
            if selezionati == dimensione_target:
                conteggio_fattibili += 1
                distribuzione_energie.append(sample.energy)
        
        tasso_fattibilita = conteggio_fattibili / len(sampleset)
        
        statistiche = {
            'tasso_fattibilita': tasso_fattibilita,
            'energia_media_fattibili': np.mean(distribuzione_energie) if distribuzione_energie else None,
            'std_energia_fattibili': np.std(distribuzione_energie) if distribuzione_energie else None,
            'distribuzione_dimensioni': np.bincount(distribuzione_dimensioni),
            'soluzioni_uniche': len(set(str(s.sample) for s in sampleset))
        }
        
        return statistiche
    
    def salva_istanza(self, istanza: Dict, nome_file: str):
        """
        Salva istanza su file pickle.
        
        Parametri:
            istanza: Dizionario istanza
            nome_file: Percorso file output
        """
        with open(nome_file, 'wb') as file:
            pickle.dump(istanza, file)
        logger.info(f"Istanza salvata in '{nome_file}'")
    
    def carica_istanza(self, nome_file: str) -> Dict:
        """
        Carica istanza da file pickle.
        
        Parametri:
            nome_file: Percorso file input
        
        Ritorna:
            Dict istanza caricata
        """
        with open(nome_file, 'rb') as file:
            istanza = pickle.load(file)
        logger.info("Istanza caricata con successo.")
        return istanza
