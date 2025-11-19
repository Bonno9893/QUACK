"""
Modulo di Ottimizzazione del Parametro Lambda

Questo modulo fornisce funzionalità per ottimizzare il parametro di penalità lambda
nelle formulazioni QUBO. Il parametro lambda controlla la forza dell'enforcement
dei vincoli ed è cruciale per ottenere soluzioni fattibili.

Il processo di ottimizzazione usa un approccio adattivo che:
1. Testa multipli valori lambda
2. Valuta fattibilità e qualità della soluzione
3. Seleziona il trade-off ottimale

Autore: Team Progetto QUACK
Data: 2024
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import json
import pandas as pd

from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

logger = logging.getLogger(__name__)


@dataclass
class ConfigurazioneRicercaLambda:
    """Configurazione per la ricerca del parametro lambda."""
    range_lambda: Tuple[float, float] = (0.1, 10.0)
    num_campioni: int = 20
    metodo_ricerca: str = 'adattivo'  # 'adattivo', 'griglia', 'binario'
    soglia_convergenza: float = 0.95  # Soglia tasso di fattibilità
    max_iterazioni: int = 50
    early_stopping: bool = True
    num_reads_per_test: int = 100  # SA reads per ogni test lambda


class OttimizzatoreLambda:
    """
    Ottimizzatore per il parametro di penalità lambda nelle formulazioni QUBO.
    
    Questa classe implementa varie strategie per trovare il valore lambda ottimale
    che bilancia la qualità della soluzione con la soddisfazione dei vincoli.
    """
    
    def __init__(
        self,
        risolutore: str = 'simulated_annealing',
        config: Optional[ConfigurazioneRicercaLambda] = None
    ):
        """
        Inizializza l'ottimizzatore lambda.
        
        Parametri:
            risolutore (str): Risolutore da usare per i test ('simulated_annealing' o 'exact')
            config (ConfigurazioneRicercaLambda, opzionale): Configurazione di ricerca
        """
        self.tipo_risolutore = risolutore
        self.config = config or ConfigurazioneRicercaLambda()
        
        # Inizializza risolutore di test
        if risolutore == 'simulated_annealing':
            self.test_sampler = SimulatedAnnealingSampler()
        else:
            raise ValueError(f"Risolutore non supportato: {risolutore}")
        
        # Cache per valori lambda testati
        self.cache_lambda = {}
        
        logger.info(f"Ottimizzatore lambda inizializzato con risolutore {risolutore}")
    
    def ottimizza_per_istanza(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        punti_candidati: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Trova lambda ottimale per una specifica istanza del problema.
        
        Parametri:
            matrice_distanze (np.ndarray): Matrice distanze a coppie
            cluster_seed (np.ndarray): Indici cluster seed
            n_espandere (int): Numero di punti da espandere
            punti_candidati (np.ndarray, opzionale): Punti candidati
        
        Ritorna:
            Tuple[float, Dict]: (lambda_ottimale, metriche_ottimizzazione)
        """
        logger.info(f"Ottimizzazione lambda per istanza: dimensione_seed={len(cluster_seed)}, "
                   f"espansione={n_espandere}")
        
        # Seleziona metodo di ottimizzazione
        if self.config.metodo_ricerca == 'adattivo':
            return self._ricerca_adattiva(
                matrice_distanze, cluster_seed, n_espandere, punti_candidati
            )
        elif self.config.metodo_ricerca == 'griglia':
            return self._ricerca_griglia(
                matrice_distanze, cluster_seed, n_espandere, punti_candidati
            )
        elif self.config.metodo_ricerca == 'binario':
            return self._ricerca_binaria(
                matrice_distanze, cluster_seed, n_espandere, punti_candidati
            )
        else:
            raise ValueError(f"Metodo di ricerca sconosciuto: {self.config.metodo_ricerca}")
    
    def _ricerca_adattiva(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        punti_candidati: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Ricerca lambda adattiva che si aggiusta basandosi sulle caratteristiche del problema.
        
        Questo metodo parte con una stima basata sulla scala delle distanze e
        raffina il valore lambda basandosi sul feedback di fattibilità.
        """
        # Stima lambda iniziale basato sulla scala delle distanze
        scala_distanze = self._stima_scala_distanze(matrice_distanze)
        lambda_corrente = scala_distanze
        
        logger.info(f"Inizio ricerca adattiva con λ iniziale = {lambda_corrente:.3f}")
        
        # Traccia valori testati e risultati
        lambda_testati = []
        risultati = []
        
        # Fase 1: Trova regione fattibile
        fattibile_trovato = False
        iterazione = 0
        
        while iterazione < self.config.max_iterazioni:
            # Testa lambda corrente
            metriche = self._test_lambda(
                lambda_corrente,
                matrice_distanze,
                cluster_seed,
                n_espandere,
                punti_candidati
            )
            
            lambda_testati.append(lambda_corrente)
            risultati.append(metriche)
            
            logger.debug(f"Iterazione {iterazione}: λ={lambda_corrente:.3f}, "
                        f"fattibilità={metriche['tasso_fattibilita']:.2%}")
            
            # Controlla se abbiamo trovato un buon lambda
            if metriche['tasso_fattibilita'] >= self.config.soglia_convergenza:
                fattibile_trovato = True
                if self.config.early_stopping:
                    break
            
            # Aggiusta lambda basato sulla fattibilità
            if metriche['media_selezionati'] < n_espandere:
                # Troppo pochi punti selezionati, diminuisci penalità
                lambda_corrente *= 0.8
            elif metriche['media_selezionati'] > n_espandere:
                # Troppi punti selezionati, aumenta penalità
                lambda_corrente *= 1.2
            else:
                # Numero giusto ma non consistente, fine-tuning
                if metriche['tasso_fattibilita'] < 0.5:
                    lambda_corrente *= 1.1
                else:
                    lambda_corrente *= 0.95
            
            # Mantieni entro i limiti
            lambda_corrente = np.clip(
                lambda_corrente,
                self.config.range_lambda[0],
                self.config.range_lambda[1]
            )
            
            iterazione += 1
        
        # Fase 2: Raffina tra lambda fattibili
        if fattibile_trovato:
            indici_fattibili = [
                i for i, r in enumerate(risultati)
                if r['tasso_fattibilita'] >= 0.8
            ]
            
            if indici_fattibili:
                # Tra i fattibili, scegli con miglior qualità soluzione
                indice_migliore = min(
                    indici_fattibili,
                    key=lambda i: risultati[i]['energia_media']
                )
                lambda_ottimale = lambda_testati[indice_migliore]
                metriche_migliori = risultati[indice_migliore]
            else:
                # Usa quello con più alta fattibilità
                indice_migliore = max(
                    range(len(risultati)),
                    key=lambda i: risultati[i]['tasso_fattibilita']
                )
                lambda_ottimale = lambda_testati[indice_migliore]
                metriche_migliori = risultati[indice_migliore]
        else:
            # Nessun lambda fattibile trovato, ritorna miglior tentativo
            logger.warning("Nessun lambda fattibile trovato nella ricerca adattiva")
            indice_migliore = max(
                range(len(risultati)),
                key=lambda i: risultati[i]['tasso_fattibilita']
            )
            lambda_ottimale = lambda_testati[indice_migliore]
            metriche_migliori = risultati[indice_migliore]
        
        # Compila report ottimizzazione
        metriche_ottimizzazione = {
            'lambda_ottimale': lambda_ottimale,
            'tasso_fattibilita': metriche_migliori['tasso_fattibilita'],
            'energia_media': metriche_migliori['energia_media'],
            'iterazioni': iterazione,
            'lambda_testati': lambda_testati,
            'tutti_risultati': risultati
        }
        
        logger.info(f"Ricerca adattiva completa: λ ottimale = {lambda_ottimale:.3f}, "
                   f"fattibilità = {metriche_migliori['tasso_fattibilita']:.2%}")
        
        return lambda_ottimale, metriche_ottimizzazione
    
    def _ricerca_griglia(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        punti_candidati: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Ricerca a griglia sui valori lambda.
        
        Testa valori lambda equispaziati nel range specificato.
        """
        valori_lambda = np.linspace(
            self.config.range_lambda[0],
            self.config.range_lambda[1],
            self.config.num_campioni
        )
        
        logger.info(f"Ricerca griglia: test {len(valori_lambda)} valori lambda")
        
        risultati = []
        for i, val_lambda in enumerate(valori_lambda):
            logger.debug(f"Test λ = {val_lambda:.3f} ({i+1}/{len(valori_lambda)})")
            
            metriche = self._test_lambda(
                val_lambda,
                matrice_distanze,
                cluster_seed,
                n_espandere,
                punti_candidati
            )
            
            risultati.append({
                'lambda': val_lambda,
                **metriche
            })
            
            # Early stopping se fattibilità perfetta trovata
            if self.config.early_stopping and metriche['tasso_fattibilita'] >= 0.99:
                logger.info(f"Early stopping: fattibilità perfetta a λ = {val_lambda:.3f}")
                break
        
        # Seleziona miglior lambda
        # Prima filtra per fattibilità, poi per energia
        risultati_fattibili = [
            r for r in risultati 
            if r['tasso_fattibilita'] >= 0.5
        ]
        
        if risultati_fattibili:
            risultato_migliore = min(risultati_fattibili, key=lambda r: r['energia_media'])
        else:
            # Nessuna soluzione fattibile, prendi miglior fattibilità
            risultato_migliore = max(risultati, key=lambda r: r['tasso_fattibilita'])
        
        lambda_ottimale = risultato_migliore['lambda']
        
        metriche_ottimizzazione = {
            'lambda_ottimale': lambda_ottimale,
            'tasso_fattibilita': risultato_migliore['tasso_fattibilita'],
            'energia_media': risultato_migliore['energia_media'],
            'dimensione_griglia': len(valori_lambda),
            'tutti_risultati': risultati
        }
        
        logger.info(f"Ricerca griglia completa: λ ottimale = {lambda_ottimale:.3f}")
        
        return lambda_ottimale, metriche_ottimizzazione
    
    def _ricerca_binaria(
        self,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        punti_candidati: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Ricerca binaria per valore lambda ottimale.
        
        Questo metodo esegue ricerca binaria per trovare il valore lambda dove
        la soluzione transita da non fattibile a fattibile.
        """
        lambda_basso = self.config.range_lambda[0]
        lambda_alto = self.config.range_lambda[1]
        
        logger.info(f"Ricerca binaria: λ ∈ [{lambda_basso:.3f}, {lambda_alto:.3f}]")
        
        lambda_testati = []
        risultati = []
        iterazione = 0
        
        while iterazione < self.config.max_iterazioni and (lambda_alto - lambda_basso) > 0.01:
            lambda_medio = (lambda_basso + lambda_alto) / 2
            
            metriche = self._test_lambda(
                lambda_medio,
                matrice_distanze,
                cluster_seed,
                n_espandere,
                punti_candidati
            )
            
            lambda_testati.append(lambda_medio)
            risultati.append({'lambda': lambda_medio, **metriche})
            
            logger.debug(f"Ricerca binaria iterazione {iterazione}: λ={lambda_medio:.3f}, "
                        f"media_selezionati={metriche['media_selezionati']:.1f}/{n_espandere}")
            
            # Aggiusta range di ricerca basato sul comportamento di selezione
            if metriche['media_selezionati'] < n_espandere:
                # Troppo pochi selezionati, serve penalità più debole
                lambda_alto = lambda_medio
            elif metriche['media_selezionati'] > n_espandere:
                # Troppi selezionati, serve penalità più forte
                lambda_basso = lambda_medio
            else:
                # Numero giusto selezionati, controlla consistenza
                if metriche['tasso_fattibilita'] < 0.9:
                    # Serve fine-tuning
                    if iterazione % 2 == 0:
                        lambda_basso = lambda_medio * 0.99
                    else:
                        lambda_alto = lambda_medio * 1.01
                else:
                    # Trovato buon lambda
                    break
            
            iterazione += 1
        
        # Seleziona migliore dai valori testati
        if risultati:
            risultati_fattibili = [
                r for r in risultati 
                if r['tasso_fattibilita'] >= 0.5
            ]
            
            if risultati_fattibili:
                risultato_migliore = min(risultati_fattibili, key=lambda r: r['energia_media'])
                lambda_ottimale = risultato_migliore['lambda']
            else:
                risultato_migliore = max(risultati, key=lambda r: r['tasso_fattibilita'])
                lambda_ottimale = risultato_migliore['lambda']
        else:
            lambda_ottimale = (lambda_basso + lambda_alto) / 2
            risultato_migliore = {'tasso_fattibilita': 0, 'energia_media': float('inf')}
        
        metriche_ottimizzazione = {
            'lambda_ottimale': lambda_ottimale,
            'tasso_fattibilita': risultato_migliore['tasso_fattibilita'],
            'energia_media': risultato_migliore['energia_media'],
            'iterazioni': iterazione,
            'lambda_testati': lambda_testati
        }
        
        logger.info(f"Ricerca binaria completa: λ ottimale = {lambda_ottimale:.3f}")
        
        return lambda_ottimale, metriche_ottimizzazione
    
    def _test_lambda(
        self,
        valore_lambda: float,
        matrice_distanze: np.ndarray,
        cluster_seed: np.ndarray,
        n_espandere: int,
        punti_candidati: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Testa un valore lambda specifico e ritorna metriche.
        
        Parametri:
            valore_lambda: Lambda da testare
            matrice_distanze: Matrice distanze
            cluster_seed: Cluster seed
            n_espandere: Dimensione espansione
            punti_candidati: Candidati
        
        Ritorna:
            Dict con metriche del test
        """
        # Controlla cache prima
        chiave_cache = (
            valore_lambda,
            hash(matrice_distanze.tobytes()),
            hash(cluster_seed.tobytes()),
            n_espandere
        )
        
        if chiave_cache in self.cache_lambda:
            return self.cache_lambda[chiave_cache]
        
        # Crea QUBO con lambda dato
        from src.ottimizzazione.formulazione_qubo import FormulatoreQUBO
        formulatore = FormulatoreQUBO(matrice_distanze)
        
        bqm = formulatore.crea_qubo_espansione_cluster(
            cluster_seed=cluster_seed,
            dimensione_target=n_espandere,
            penalita_lambda=valore_lambda,
            punti_candidati=punti_candidati
        )
        
        # Esegui test solver
        sampleset = self.test_sampler.sample(
            bqm,
            num_reads=self.config.num_reads_per_test,
            num_sweeps=100,  # Test veloce
            seed=42
        )
        
        # Analizza risultati
        conteggio_fattibili = 0
        totale_selezionati = []
        energie = []
        
        for sample in sampleset:
            selezionati = sum(sample.sample.values())
            totale_selezionati.append(selezionati)
            energie.append(sample.energy)
            
            if selezionati == n_espandere:
                conteggio_fattibili += sample.num_occurrences
        
        occorrenze_totali = sum(s.num_occurrences for s in sampleset)
        tasso_fattibilita = conteggio_fattibili / occorrenze_totali if occorrenze_totali > 0 else 0
        
        metriche = {
            'tasso_fattibilita': tasso_fattibilita,
            'media_selezionati': np.mean(totale_selezionati),
            'std_selezionati': np.std(totale_selezionati),
            'energia_media': np.mean(energie),
            'energia_min': min(energie),
            'soluzioni_uniche': len(set(str(s.sample) for s in sampleset))
        }
        
        # Cache risultato
        self.cache_lambda[chiave_cache] = metriche
        
        return metriche
    
    def _stima_scala_distanze(self, matrice_distanze: np.ndarray) -> float:
        """
        Stima scala lambda appropriata basata sulla matrice distanze.
        
        Parametri:
            matrice_distanze: Distanze a coppie
        
        Ritorna:
            float: Scala lambda stimata
        """
        # Ottieni distanze non-zero
        distanze = matrice_distanze[np.triu_indices_from(matrice_distanze, k=1)]
        distanze = distanze[distanze > 0]
        
        if len(distanze) == 0:
            return 1.0
        
        # Usa distanza mediana come stima di scala
        # Lambda dovrebbe essere comparabile ai valori tipici di distanza
        distanza_mediana = np.median(distanze)
        
        # Aggiusta basato sulla dimensione del problema (problemi più grandi necessitano penalità più forti)
        n_punti = matrice_distanze.shape[0]
        fattore_dimensione = 1 + np.log10(n_punti)
        
        lambda_stimato = distanza_mediana * fattore_dimensione
        
        # Mantieni entro range configurato
        lambda_stimato = np.clip(
            lambda_stimato,
            self.config.range_lambda[0],
            self.config.range_lambda[1]
        )
        
        logger.debug(f"Scala λ stimata: {lambda_stimato:.3f} "
                    f"(dist mediana: {distanza_mediana:.3f}, fattore dimensione: {fattore_dimensione:.2f})")
        
        return lambda_stimato
    
    def ottimizzazione_batch_lambda(
        self,
        istanze: List[Dict],
        percorso_output: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ottimizza lambda per batch di istanze.
        
        Questo metodo è basato sul notebook Server_batch_processing_for_lambda2_OP.ipynb
        
        Parametri:
            istanze: Lista di istanze da processare
            percorso_output: Percorso per salvare risultati Excel
        
        Ritorna:
            DataFrame con risultati ottimizzazione
        """
        risultati = []
        
        for istanza in istanze:
            logger.info(f"Processando istanza {istanza['id']}")
            
            # Estrai parametri istanza
            matrice_distanze = istanza['matrice_distanze']
            cluster_seed = istanza['cluster_seed']
            n_espandere = istanza['n_espandere']
            
            # Ottimizza lambda
            lambda_ottimale, metriche = self.ottimizza_per_istanza(
                matrice_distanze=matrice_distanze,
                cluster_seed=cluster_seed,
                n_espandere=n_espandere
            )
            
            # Registra risultati
            risultato = {
                'istanza_id': istanza['id'],
                'lambda_ottimale': lambda_ottimale,
                'tasso_fattibilita': metriche['tasso_fattibilita'],
                'energia_media': metriche['energia_media'],
                'iterazioni': metriche.get('iterazioni', 0),
                'n_punti': matrice_distanze.shape[0],
                'dimensione_seed': len(cluster_seed),
                'n_espandere': n_espandere
            }
            
            risultati.append(risultato)
        
        # Crea DataFrame
        df_risultati = pd.DataFrame(risultati)
        
        # Salva in Excel se richiesto
        if percorso_output:
            df_risultati.to_excel(percorso_output, index=False)
            logger.info(f"Risultati salvati in {percorso_output}")
        
        return df_risultati
    
    def salva_storico_ottimizzazione(self, percorso_file: str):
        """
        Salva storico ottimizzazione su file.
        
        Parametri:
            percorso_file: Percorso per salvare file JSON
        """
        storico = {
            'config': {
                'range_lambda': self.config.range_lambda,
                'num_campioni': self.config.num_campioni,
                'metodo_ricerca': self.config.metodo_ricerca
            },
            'dimensione_cache': len(self.cache_lambda),
            'risultati_cache': [
                {
                    'lambda': k[0],
                    'n_espandere': k[3],
                    'metriche': v
                }
                for k, v in list(self.cache_lambda.items())[:100]  # Limita a 100 entry
            ]
        }
        
        with open(percorso_file, 'w') as f:
            json.dump(storico, f, indent=2, default=str)
        
        logger.info(f"Storico ottimizzazione salvato in {percorso_file}")
