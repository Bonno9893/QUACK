"""
Modulo Risolutore D-Wave Quantum Annealing

Questo modulo fornisce l'interfaccia per risolvere problemi di clustering usando
l'hardware di quantum annealing di D-Wave. Gestisce la pipeline completa dalla
formulazione QUBO all'esecuzione quantistica e al processamento dei risultati.

Caratteristiche Principali:
    - Embedding automatico alla topologia dell'hardware quantistico
    - Parametri di annealing configurabili
    - Ottimizzazione della forza delle catene
    - Post-processamento e validazione dei risultati
    - Raccolta delle metriche di performance

Autore: Team Progetto QUACK
Data: 2024
Licenza: MIT
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.embedding import find_embedding
import minorminer

# Configurazione del logging
logger = logging.getLogger(__name__)


@dataclass
class ConfigurazioneDWave:
    """Parametri di configurazione per il risolutore D-Wave."""
    token_api: str
    nome_risolutore: Optional[str] = None
    num_letture: int = 1000
    tempo_annealing: int = 20  # microsecondi
    forza_catena: Optional[float] = None
    auto_scala: bool = True
    usa_embedding_fisso: bool = False
    topologia: str = "zephyr"  # pegasus, zephyr, o chimera


@dataclass
class SoluzioneQuantistica:
    """Contenitore per la soluzione del quantum annealing e metadati."""
    miglior_campione: Dict[str, int]
    energia: float
    num_occorrenze: int
    è_fattibile: bool
    indici_selezionati: np.ndarray
    info_tempistiche: Dict[str, float]
    info_embedding: Dict[str, Any]
    tutti_campioni: Optional[List[Dict]] = None


class RisolutoreClusteringDWave:
    """
    Una classe risolutore per problemi di clustering usando il quantum annealing D-Wave.
    
    Questa classe gestisce il workflow completo per risolvere problemi di clustering
    su hardware quantistico, inclusi embedding, tuning dei parametri, e interpretazione
    dei risultati.
    """
    
    def __init__(self, config: ConfigurazioneDWave):
        """
        Inizializza il risolutore D-Wave con la configurazione.
        
        Parametri:
            config (ConfigurazioneDWave): Oggetto di configurazione con parametri D-Wave
            
        Solleva:
            ConnectionError: Se non riesce a connettersi al servizio cloud D-Wave
        """
        self.config = config
        self.sampler = None
        self.sampler_composito = None
        self.embedding_fisso = None
        
        # Inizializza connessione a D-Wave
        self._inizializza_sampler()
        
        logger.info(f"Risolutore D-Wave inizializzato con {self.sampler.properties['chip_id']}")
        
    def _inizializza_sampler(self) -> None:
        """
        Inizializza il sampler D-Wave e il composito di embedding.
        
        Questo metodo stabilisce la connessione all'hardware quantistico e
        configura la strategia di embedding (fissa o dinamica).
        """
        try:
            # Connetti al quantum annealer D-Wave
            if self.config.nome_risolutore:
                self.sampler = DWaveSampler(
                    token=self.config.token_api,
                    solver=self.config.nome_risolutore
                )
            else:
                # Auto-selezione del risolutore basata sulla preferenza di topologia
                caratteristiche_solver = {'topology__type': self.config.topologia}
                self.sampler = DWaveSampler(
                    token=self.config.token_api,
                    solver=caratteristiche_solver
                )
            
            # Configura il composito di embedding
            if self.config.usa_embedding_fisso:
                # Per problemi ripetuti, usa embedding fisso per consistenza
                self.sampler_composito = FixedEmbeddingComposite(self.sampler, {})
            else:
                # Embedding dinamico per ogni problema
                self.sampler_composito = EmbeddingComposite(self.sampler)
                
            # Registra proprietà hardware
            proprietà = self.sampler.properties
            logger.info(f"Connesso a: {proprietà['chip_id']}")
            logger.info(f"Topologia: {proprietà['topology']['type']}")
            logger.info(f"Qubit: {proprietà['num_qubits']}")
            logger.info(f"Accoppiatori: {len(proprietà['couplers'])}")
            
        except Exception as e:
            logger.error(f"Fallita inizializzazione del sampler D-Wave: {e}")
            raise ConnectionError(f"Impossibile connettersi a D-Wave: {e}")
    
    def risolvi_espansione_cluster(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        penalita_lambda: float,
        punti_candidati: Optional[np.ndarray] = None
    ) -> SoluzioneQuantistica:
        """
        Risolvi il problema di espansione del cluster usando il quantum annealing.
        
        Questo metodo prende un cluster seed e lo espande selezionando esattamente
        n_espandere punti aggiuntivi che minimizzano la distanza intra-cluster totale.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice delle distanze a coppie
            cluster_seed (np.ndarray): Indici dei punti nel cluster seed
            n_espandere (int): Numero di punti da aggiungere al cluster
            penalita_lambda (float): Parametro di penalità per il vincolo di cardinalità
            punti_candidati (np.ndarray, opzionale): Punti candidati per l'espansione
        
        Ritorna:
            SoluzioneQuantistica: Oggetto soluzione contenente risultati e metadati
            
        Esempio:
            >>> risolutore = RisolutoreClusteringDWave(config)
            >>> soluzione = risolutore.risolvi_espansione_cluster(
            ...     matrice_distanze=mat_dist,
            ...     cluster_seed=np.array([0, 1, 2]),
            ...     n_espandere=5,
            ...     penalita_lambda=2.0
            ... )
            >>> print(f"Punti selezionati: {soluzione.indici_selezionati}")
        """
        logger.info(f"Risoluzione espansione cluster: dimensione seed={len(cluster_seed)}, "
                   f"espandi di={n_espandere}, λ={penalita_lambda:.3f}")
        
        # Crea formulazione QUBO
        from ..ottimizzazione.formulazione_qubo import FormulatoreQUBO
        formulatore = FormulatoreQUBO(matrice_distanze)
        
        bqm = formulatore.crea_qubo_espansione_cluster(
            cluster_seed=cluster_seed,
            dimensione_target=n_espandere,
            penalita_lambda=penalita_lambda,
            punti_candidati=punti_candidati
        )
        
        # Risolvi su hardware quantistico
        soluzione = self._risolvi_bqm(bqm, n_espandere)
        
        # Post-processa per ottenere gli indici dei punti effettivi
        if punti_candidati is not None:
            variabili_selezionate = [var for var, val in soluzione.miglior_campione.items() if val == 1]
            indici_selezionati = [punti_candidati[int(var.split('_')[1])] for var in variabili_selezionate]
            soluzione.indici_selezionati = np.array(indici_selezionati)
        
        return soluzione
    
    def _risolvi_bqm(
        self,
        bqm: BinaryQuadraticModel,
        dimensione_target: int
    ) -> SoluzioneQuantistica:
        """
        Risolvi un Modello Quadratico Binario su hardware D-Wave.
        
        Questo metodo interno gestisce il processo effettivo di quantum annealing,
        inclusi calcolo della forza delle catene, embedding, ed estrazione dei risultati.
        
        Parametri:
            bqm (BinaryQuadraticModel): Il modello QUBO da risolvere
            dimensione_target (int): Numero atteso di variabili selezionate
        
        Ritorna:
            SoluzioneQuantistica: Soluzione processata con metadati
        """
        # Calcola la forza delle catene se non fornita
        if self.config.forza_catena is None:
            forza_catena = self._calcola_forza_catena(bqm)
        else:
            forza_catena = self.config.forza_catena
        
        logger.info(f"Usando forza catena: {forza_catena:.3f}")
        
        # Prepara parametri di campionamento
        parametri_campionamento = {
            'num_reads': self.config.num_letture,
            'annealing_time': self.config.tempo_annealing,
            'chain_strength': forza_catena,
            'return_embedding': True,
            'answer_mode': 'histogram'  # Ottieni risultati aggregati
        }
        
        # Esegui quantum annealing
        tempo_inizio = time.time()
        try:
            sampleset = self.sampler_composito.sample(bqm, **parametri_campionamento)
        except Exception as e:
            logger.error(f"Quantum annealing fallito: {e}")
            raise RuntimeError(f"Errore esecuzione D-Wave: {e}")
        
        tempo_fine = time.time()
        
        # Estrai informazioni temporali
        info_tempistiche = self._estrai_info_tempistiche(sampleset)
        info_tempistiche['tempo_totale'] = tempo_fine - tempo_inizio
        
        # Estrai informazioni di embedding
        info_embedding = self._estrai_info_embedding(sampleset)
        
        # Ottieni la miglior soluzione
        miglior_campione = sampleset.first.sample
        miglior_energia = sampleset.first.energy
        migliori_occorrenze = sampleset.first.num_occurrences
        
        # Valida fattibilità della soluzione
        conteggio_selezionati = sum(miglior_campione.values())
        è_fattibile = (conteggio_selezionati == dimensione_target)
        
        # Estrai indici selezionati
        variabili_selezionate = [var for var, val in miglior_campione.items() if val == 1]
        indici_selezionati = np.array([int(var.split('_')[1]) for var in variabili_selezionate])
        
        # Registra qualità della soluzione
        logger.info(f"Miglior soluzione: energia={miglior_energia:.3f}, "
                   f"occorrenze={migliori_occorrenze}/{self.config.num_letture}, "
                   f"fattibile={è_fattibile}")
        
        # Compila tutti i campioni se richiesto
        tutti_campioni = None
        if hasattr(sampleset, 'record'):
            tutti_campioni = [
                {
                    'campione': dict(zip(sampleset.variables, sample)),
                    'energia': energy,
                    'num_occorrenze': num_occ
                }
                for sample, energy, num_occ in sampleset.record
            ]
        
        return SoluzioneQuantistica(
            miglior_campione=miglior_campione,
            energia=miglior_energia,
            num_occorrenze=migliori_occorrenze,
            è_fattibile=è_fattibile,
            indici_selezionati=indici_selezionati,
            info_tempistiche=info_tempistiche,
            info_embedding=info_embedding,
            tutti_campioni=tutti_campioni
        )
    
    def _calcola_forza_catena(self, bqm: BinaryQuadraticModel) -> float:
        """
        Calcola la forza catena appropriata per il problema.
        
        La forza catena deve essere abbastanza grande per mantenere l'integrità
        delle catene ma non così grande da dominare la scala energetica del problema.
        
        Parametri:
            bqm (BinaryQuadraticModel): Il modello QUBO
        
        Ritorna:
            float: Forza catena calcolata
        """
        # Ottieni il range dei coefficienti nel BQM
        range_lineare = 0
        if bqm.linear:
            valori_lineari = list(bqm.linear.values())
            range_lineare = max(valori_lineari) - min(valori_lineari)
        
        range_quadratico = 0
        if bqm.quadratic:
            valori_quadratici = list(bqm.quadratic.values())
            range_quadratico = max(valori_quadratici) - min(valori_quadratici)
        
        # La forza catena dovrebbe essere maggiore della scala energetica del problema
        # Euristica comune: 1.5-2x la magnitudine massima del coefficiente
        coeff_max = max(range_lineare, range_quadratico)
        
        # Aggiungi un buffer per i termini di vincolo (che sono tipicamente più grandi)
        forza_catena = 1.5 * coeff_max
        
        # Assicura forza catena minima
        forza_catena = max(forza_catena, 1.0)
        
        logger.info(f"Forza catena calcolata: {forza_catena:.3f} "
                   f"(range lineare: {range_lineare:.3f}, "
                   f"range quadratico: {range_quadratico:.3f})")
        
        return forza_catena
    
    def _estrai_info_tempistiche(self, sampleset) -> Dict[str, float]:
        """
        Estrai informazioni dettagliate sui tempi dal sampleset.
        
        Parametri:
            sampleset: Oggetto sampleset D-Wave
        
        Ritorna:
            Dict[str, float]: Breakdown dei tempi in secondi
        """
        tempistiche = {}
        
        if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
            timing_dwave = sampleset.info['timing']
            
            # Converti microsecondi in secondi per i campi rilevanti
            campi_timing = {
                'qpu_access_time': 1e-6,  # Tempo totale di accesso QPU
                'qpu_programming_time': 1e-6,  # Tempo per programmare la QPU
                'qpu_sampling_time': 1e-6,  # Tempo effettivo di annealing
                'qpu_anneal_time_per_sample': 1e-6,  # Tempo annealing per campione
                'qpu_readout_time_per_sample': 1e-6,  # Tempo lettura per campione
                'qpu_delay_time_per_sample': 1e-6,  # Delay tra campioni
                'post_processing_overhead_time': 1e-6,  # Tempo post-processing
                'total_post_processing_time': 1e-6  # Post-processing totale
            }
            
            for campo, conversione in campi_timing.items():
                if campo in timing_dwave:
                    tempistiche[campo] = timing_dwave[campo] * conversione
            
            # Registra breakdown dei tempi
            logger.debug("Breakdown tempi QPU:")
            for chiave, valore in tempistiche.items():
                logger.debug(f"  {chiave}: {valore:.6f} secondi")
        
        return tempistiche
    
    def _estrai_info_embedding(self, sampleset) -> Dict[str, Any]:
        """
        Estrai informazioni di embedding dal sampleset.
        
        Parametri:
            sampleset: Oggetto sampleset D-Wave
        
        Ritorna:
            Dict[str, Any]: Statistiche di embedding
        """
        info_embedding = {}
        
        if hasattr(sampleset, 'info') and 'embedding_context' in sampleset.info:
            contesto = sampleset.info['embedding_context']
            
            if 'embedding' in contesto:
                embedding = contesto['embedding']
                
                # Calcola statistiche di embedding
                lunghezze_catene = [len(catena) for catena in embedding.values()]
                
                info_embedding = {
                    'num_variabili_logiche': len(embedding),
                    'num_qubit_fisici': sum(lunghezze_catene),
                    'lunghezza_catena_max': max(lunghezze_catene) if lunghezze_catene else 0,
                    'lunghezza_catena_media': np.mean(lunghezze_catene) if lunghezze_catene else 0,
                    'lunghezza_catena_min': min(lunghezze_catene) if lunghezze_catene else 0
                }
                
                # Controlla rotture di catena
                if 'chain_break_fraction' in contesto:
                    info_embedding['frazione_rotture_catena'] = contesto['chain_break_fraction']
                
                logger.debug(f"Statistiche embedding: {info_embedding}")
        
        return info_embedding
    
    def risolvi_batch(
        self,
        problemi: List[Dict[str, Any]],
        parallelo: bool = False
    ) -> List[SoluzioneQuantistica]:
        """
        Risolvi più problemi di clustering in batch.
        
        Questo metodo può opzionalmente riusare gli embedding per problemi simili
        per migliorare le performance.
        
        Parametri:
            problemi (List[Dict]): Lista di specifiche dei problemi
            parallelo (bool): Se usare embedding parallelo (se disponibile)
        
        Ritorna:
            List[SoluzioneQuantistica]: Soluzioni per tutti i problemi
        """
        soluzioni = []
        
        # Se usando embedding fisso, calcolalo una volta per il primo problema
        if self.config.usa_embedding_fisso and problemi:
            primo_problema = problemi[0]
            # Genera embedding dal primo problema
            # ... (codice generazione embedding)
        
        for i, problema in enumerate(problemi):
            logger.info(f"Risoluzione problema {i+1}/{len(problemi)}")
            
            soluzione = self.risolvi_espansione_cluster(
                matrice_distanze=problema['matrice_distanze'],
                cluster_seed=problema.get('cluster_seed', np.array([])),
                n_espandere=problema['n_espandere'],
                penalita_lambda=problema['penalita_lambda']
            )
            
            soluzioni.append(soluzione)
        
        return soluzioni
    
    def ottimizza_schedule_annealing(
        self,
        bqm: BinaryQuadraticModel,
        schedule_test: Optional[List[List[Tuple[float, float]]]] = None
    ) -> Dict[str, Any]:
        """
        Sperimenta con diversi schedule di annealing per trovare i parametri ottimali.
        
        Parametri:
            bqm (BinaryQuadraticModel): Problema da testare
            schedule_test (List): Schedule di annealing personalizzati da testare
        
        Ritorna:
            Dict[str, Any]: Miglior schedule e metriche di performance
        """
        if schedule_test is None:
            # Schedule di default da testare
            schedule_test = [
                [(0.0, 0.0), (20.0, 1.0)],  # Lineare standard
                [(0.0, 0.0), (10.0, 0.5), (20.0, 1.0)],  # Pausa nel mezzo
                [(0.0, 0.0), (5.0, 0.8), (20.0, 1.0)],  # Rampa veloce
            ]
        
        miglior_schedule = None
        miglior_energia = float('inf')
        risultati = []
        
        for schedule in schedule_test:
            logger.info(f"Test schedule: {schedule}")
            
            # Campiona con schedule personalizzato
            sampleset = self.sampler.sample(
                bqm,
                num_reads=100,
                anneal_schedule=schedule
            )
            
            # Registra risultati
            energia_media = np.mean([s.energy for s in sampleset])
            energia_min = sampleset.first.energy
            
            risultati.append({
                'schedule': schedule,
                'energia_media': energia_media,
                'energia_min': energia_min
            })
            
            if energia_min < miglior_energia:
                miglior_energia = energia_min
                miglior_schedule = schedule
        
        return {
            'miglior_schedule': miglior_schedule,
            'miglior_energia': miglior_energia,
            'tutti_risultati': risultati
        }
