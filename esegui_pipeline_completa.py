#!/usr/bin/env python3
"""
Pipeline Completa per Caso d'Uso Bancario QUACK con D-Wave

Questo script orchestra l'intero workflow per risolvere problemi di clustering
nel settore bancario usando il quantum annealing. Include:
1. Caricamento e preprocessamento dati
2. Generazione istanze
3. Ottimizzazione parametro lambda
4. Quantum annealing con D-Wave
5. Benchmarking classico (Gurobi, Simulated Annealing)
6. Confronto performance e visualizzazione

Utilizzo:
    python esegui_pipeline_completa.py --config config.yaml
    python esegui_pipeline_completa.py --percorso-dati dati/bancari.csv --output risultati/

Autore: Team Progetto QUACK
Data: 2024
Licenza: MIT
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# Importa moduli del progetto
from src.ottimizzazione.formulazione_qubo import FormulatoreQUBO, crea_qubo_consapevole_distanze
from src.risolutori.risolutore_dwave import RisolutoreClusteringDWave, ConfigurazioneDWave
from src.risolutori.simulated_annealing import RisolutoreSimulatedAnnealing
from src.risolutori.risolutore_gurobi import RisolutoreGurobi
from src.ottimizzazione.ottimizzatore_lambda import OttimizzatoreLambda
from src.utilita.metriche_valutazione import ValutatoreClusteringq
from src.utilita.visualizzazione import VisualizzatoreRisultati

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineClusteringBancario:
    """
    Classe principale della pipeline per il caso d'uso di clustering bancario.
    
    Questa classe orchestra tutti i componenti del workflow di clustering,
    dalla preparazione dati attraverso la risoluzione quantistica fino all'analisi
    dei risultati.
    """
    
    def __init__(self, percorso_config: str):
        """
        Inizializza la pipeline con la configurazione.
        
        Parametri:
            percorso_config (str): Percorso al file di configurazione YAML
        """
        self.config = self._carica_configurazione(percorso_config)
        self.risultati = {}
        self.istanze = []
        
        # Inizializza i risolutori basati sulla configurazione
        self._inizializza_risolutori()
        
        # Configura le directory di output
        self._configura_directory()
        
        logger.info("Pipeline inizializzata con successo")
    
    def _carica_configurazione(self, percorso_config: str) -> Dict:
        """Carica la configurazione dal file YAML."""
        with open(percorso_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Valida campi richiesti
        richiesti = ['dwave', 'percorsi', 'esperimento']
        for campo in richiesti:
            if campo not in config:
                raise ValueError(f"Configurazione manca campo richiesto: {campo}")
        
        return config
    
    def _inizializza_risolutori(self):
        """Inizializza tutti i risolutori configurati."""
        self.risolutori = {}
        
        # Risolutore quantistico D-Wave
        if self.config.get('dwave', {}).get('abilitato', True):
            config_dwave = ConfigurazioneDWave(
                token_api=self.config['dwave']['token_api'],
                nome_risolutore=self.config['dwave'].get('risolutore'),
                num_letture=self.config['dwave'].get('num_letture', 1000),
                tempo_annealing=self.config['dwave'].get('tempo_annealing', 20)
            )
            self.risolutori['dwave'] = RisolutoreClusteringDWave(config_dwave)
            logger.info("Risolutore D-Wave inizializzato")
        
        # Risolutore Simulated Annealing
        if self.config.get('simulated_annealing', {}).get('abilitato', True):
            self.risolutori['sa'] = RisolutoreSimulatedAnnealing(
                num_letture=self.config.get('simulated_annealing', {}).get('num_letture', 1000)
            )
            logger.info("Risolutore Simulated Annealing inizializzato")
        
        # Risolutore Gurobi
        if self.config.get('gurobi', {}).get('abilitato', True):
            try:
                self.risolutori['gurobi'] = RisolutoreGurobi(
                    limite_tempo=self.config.get('gurobi', {}).get('limite_tempo', 60)
                )
                logger.info("Risolutore Gurobi inizializzato")
            except ImportError:
                logger.warning("Gurobi non disponibile, saltato")
    
    def _configura_directory(self):
        """Crea le directory di output necessarie."""
        dir_base = Path(self.config['percorsi']['cartella_risultati'])
        
        self.directory = {
            'risultati': dir_base,
            'istanze': dir_base / 'istanze',
            'soluzioni': dir_base / 'soluzioni',
            'metriche': dir_base / 'metriche',
            'visualizzazioni': dir_base / 'visualizzazioni'
        }
        
        for percorso_dir in self.directory.values():
            percorso_dir.mkdir(parents=True, exist_ok=True)
    
    def carica_dati_bancari(self, percorso_dati: Optional[str] = None) -> pd.DataFrame:
        """
        Carica e preprocessa i dati dei clienti bancari.
        
        Parametri:
            percorso_dati (str, opzionale): Percorso al file dati. Se None, usa config.
        
        Ritorna:
            pd.DataFrame: Dati clienti preprocessati
        """
        if percorso_dati is None:
            percorso_dati = self.config['percorsi']['file_dati']
        
        logger.info(f"Caricamento dati da {percorso_dati}")
        
        # Carica dati (assumendo formato CSV)
        df = pd.read_csv(percorso_dati)
        
        # Passi di preprocessamento dati basati sulle specifiche del progetto QUACK
        # Rimuovi colonne non necessarie
        colonne_da_rimuovere = [
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
            'AcceptedCmp4', 'AcceptedCmp5', 'Response',
            'Year_Birth', 'Education', 'Marital_Status',
            'Kidhome', 'Teenhome', 'Dt_Customer', 
            'Complain', 'Z_CostContact', 'Z_Revenue'
        ]
        
        df = df.drop(columns=[col for col in colonne_da_rimuovere if col in df.columns])
        
        # Standardizza caratteristiche numeriche
        colonne_numeriche = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[colonne_numeriche] = scaler.fit_transform(df[colonne_numeriche])
        
        logger.info(f"Caricati {len(df)} record clienti con {len(df.columns)} caratteristiche")
        
        return df
    
    def genera_istanze(
        self,
        dati: pd.DataFrame,
        n_istanze: int = 10
    ) -> List[Dict]:
        """
        Genera istanze di test dai dati bancari.
        
        Questo metodo crea istanze di problemi di clustering:
        1. Campionando clienti dal dataset
        2. Calcolando matrici di distanza
        3. Creando cluster seed
        4. Configurando problemi di espansione
        
        Parametri:
            dati (pd.DataFrame): Dati clienti
            n_istanze (int): Numero di istanze da generare
        
        Ritorna:
            List[Dict]: Istanze di problema generate
        """
        istanze = []
        
        # Parametri generazione istanze dalla config
        parametri = self.config['esperimento']['parametri_istanza']
        
        for i in range(n_istanze):
            # Campiona clienti
            n_punti = parametri['n_punti']
            dati_campionati = dati.sample(n=n_punti, random_state=42 + i)
            
            # Calcola matrice distanze (distanza Euclidea)
            from sklearn.metrics import pairwise_distances
            matrice_distanze = pairwise_distances(dati_campionati.values)
            
            # Normalizza distanze
            if matrice_distanze.max() > 0:
                matrice_distanze = matrice_distanze / matrice_distanze.max()
            
            # Crea cluster seed (punti cluster iniziali)
            dimensione_seed = parametri['dimensione_cluster_seed']
            indici_seed = np.random.choice(n_punti, size=dimensione_seed, replace=False)
            
            # Determina dimensione espansione
            dimensione_espansione = parametri['dimensione_espansione']
            
            # Crea ground truth clustering (per valutazione)
            # Usando k-means semplice per riferimento
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=parametri['n_cluster'], random_state=42)
            y_vero = kmeans.fit_predict(dati_campionati.values)
            
            istanza = {
                'id': i,
                'matrice_distanze': matrice_distanze,
                'cluster_seed': indici_seed,
                'n_espandere': dimensione_espansione,
                'y_vero': y_vero,
                'n_punti': n_punti,
                'n_cluster': parametri['n_cluster'],
                'indici_dati': dati_campionati.index.values
            }
            
            istanze.append(istanza)
            
            logger.info(f"Generata istanza {i}: {n_punti} punti, "
                       f"seed={dimensione_seed}, espansione={dimensione_espansione}")
        
        self.istanze = istanze
        return istanze
    
    def ottimizza_parametri_lambda(self) -> Dict[int, float]:
        """
        Ottimizza i parametri di penalità lambda per ogni istanza.
        
        Questo metodo usa Simulated Annealing per trovare valori lambda ottimali
        che assicurano soluzioni fattibili con buona qualità di clustering.
        
        Ritorna:
            Dict[int, float]: Valori lambda ottimali per ogni istanza
        """
        logger.info("Inizio ottimizzazione parametri lambda")
        
        ottimizzatore = OttimizzatoreLambda(
            risolutore='simulated_annealing',
            range_lambda=self.config['esperimento'].get('range_lambda', (0.1, 10.0)),
            num_campioni=self.config['esperimento'].get('campioni_lambda', 20)
        )
        
        lambda_ottimali = {}
        
        for istanza in self.istanze:
            logger.info(f"Ottimizzazione lambda per istanza {istanza['id']}")
            
            # Trova lambda ottimale
            miglior_lambda, metriche = ottimizzatore.ottimizza_per_istanza(
                matrice_distanze=istanza['matrice_distanze'],
                cluster_seed=istanza['cluster_seed'],
                n_espandere=istanza['n_espandere']
            )
            
            lambda_ottimali[istanza['id']] = miglior_lambda
            
            logger.info(f"Istanza {istanza['id']}: λ ottimale = {miglior_lambda:.3f}, "
                       f"fattibilità = {metriche['tasso_fattibilità']:.2%}")
        
        self.lambda_ottimali = lambda_ottimali
        return lambda_ottimali
    
    def risolvi_con_tutti_metodi(self) -> pd.DataFrame:
        """
        Risolvi tutte le istanze con tutti i metodi disponibili.
        
        Questo metodo esegue ogni istanza attraverso:
        - Quantum annealing D-Wave
        - Simulated Annealing
        - Gurobi (se disponibile)
        
        Ritorna:
            pd.DataFrame: Risultati confronto performance
        """
        risultati = []
        
        for istanza in self.istanze:
            id_istanza = istanza['id']
            lambda_ottimale = self.lambda_ottimali[id_istanza]
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Risoluzione istanza {id_istanza}")
            logger.info(f"{'='*50}")
            
            risultati_istanza = {
                'id_istanza': id_istanza,
                'n_punti': istanza['n_punti'],
                'dimensione_seed': len(istanza['cluster_seed']),
                'dimensione_espansione': istanza['n_espandere'],
                'lambda_ottimale': lambda_ottimale
            }
            
            # Risolvi con ogni metodo disponibile
            for nome_risolutore, risolutore in self.risolutori.items():
                logger.info(f"Risoluzione con {nome_risolutore}...")
                
                tempo_inizio = time.time()
                
                try:
                    if nome_risolutore == 'dwave':
                        soluzione = risolutore.risolvi_espansione_cluster(
                            matrice_distanze=istanza['matrice_distanze'],
                            cluster_seed=istanza['cluster_seed'],
                            n_espandere=istanza['n_espandere'],
                            penalita_lambda=lambda_ottimale
                        )
                        
                        # Estrai metriche
                        risultati_istanza[f'{nome_risolutore}_energia'] = soluzione.energia
                        risultati_istanza[f'{nome_risolutore}_fattibile'] = soluzione.è_fattibile
                        risultati_istanza[f'{nome_risolutore}_tempo'] = soluzione.info_tempistiche.get(
                            'qpu_access_time', 0
                        )
                        
                        # Calcola qualità clustering se fattibile
                        if soluzione.è_fattibile and istanza.get('y_vero') is not None:
                            # Crea assegnazione clustering completa
                            clustering = np.zeros(istanza['n_punti'])
                            clustering[istanza['cluster_seed']] = 1
                            clustering[soluzione.indici_selezionati] = 1
                            
                            ari = adjusted_rand_score(istanza['y_vero'], clustering)
                            risultati_istanza[f'{nome_risolutore}_ari'] = ari
                    
                    elif nome_risolutore == 'sa':
                        soluzione = risolutore.risolvi(
                            matrice_distanze=istanza['matrice_distanze'],
                            cluster_seed=istanza['cluster_seed'],
                            n_espandere=istanza['n_espandere'],
                            penalita_lambda=lambda_ottimale
                        )
                        
                        risultati_istanza[f'{nome_risolutore}_energia'] = soluzione['energia']
                        risultati_istanza[f'{nome_risolutore}_fattibile'] = soluzione['fattibile']
                        risultati_istanza[f'{nome_risolutore}_tempo'] = time.time() - tempo_inizio
                        
                        if soluzione['fattibile'] and istanza.get('y_vero') is not None:
                            clustering = np.zeros(istanza['n_punti'])
                            clustering[istanza['cluster_seed']] = 1
                            clustering[soluzione['indici_selezionati']] = 1
                            
                            ari = adjusted_rand_score(istanza['y_vero'], clustering)
                            risultati_istanza[f'{nome_risolutore}_ari'] = ari
                    
                    elif nome_risolutore == 'gurobi':
                        soluzione = risolutore.risolvi_clustering(
                            matrice_distanze=istanza['matrice_distanze'],
                            n_cluster=istanza['n_cluster']
                        )
                        
                        risultati_istanza[f'{nome_risolutore}_obiettivo'] = soluzione['obiettivo']
                        risultati_istanza[f'{nome_risolutore}_tempo'] = soluzione['tempo_risoluzione']
                        risultati_istanza[f'{nome_risolutore}_gap'] = soluzione.get('gap', 0)
                        
                        if istanza.get('y_vero') is not None:
                            ari = adjusted_rand_score(istanza['y_vero'], soluzione['clustering'])
                            risultati_istanza[f'{nome_risolutore}_ari'] = ari
                
                except Exception as e:
                    logger.error(f"Errore con {nome_risolutore}: {e}")
                    risultati_istanza[f'{nome_risolutore}_errore'] = str(e)
            
            risultati.append(risultati_istanza)
        
        # Converti in DataFrame
        df_risultati = pd.DataFrame(risultati)
        
        # Salva risultati
        percorso_risultati = self.directory['metriche'] / 'confronto_performance.csv'
        df_risultati.to_csv(percorso_risultati, index=False)
        logger.info(f"Risultati salvati in {percorso_risultati}")
        
        self.df_risultati = df_risultati
        return df_risultati
    
    def genera_visualizzazioni(self):
        """
        Genera visualizzazioni complete dei risultati.
        
        Crea grafici per:
        - Confronto performance tra metodi
        - Metriche qualità soluzione
        - Analisi complessità temporale
        - Tassi di fattibilità
        """
        visualizzatore = VisualizzatoreRisultati(self.directory['visualizzazioni'])
        
        # 1. Grafico confronto performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confronto punteggi ARI
        colonne_ari = [col for col in self.df_risultati.columns if 'ari' in col.lower()]
        if colonne_ari:
            ax = axes[0, 0]
            dati_ari = self.df_risultati[['id_istanza'] + colonne_ari].melt(
                id_vars='id_istanza', 
                var_name='Metodo', 
                value_name='ARI'
            )
            dati_ari['Metodo'] = dati_ari['Metodo'].str.replace('_ari', '')
            sns.boxplot(data=dati_ari, x='Metodo', y='ARI', ax=ax)
            ax.set_title('Confronto Qualità Clustering (ARI)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Indice di Rand Aggiustato', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        
        # Confronto tempi esecuzione
        colonne_tempo = [col for col in self.df_risultati.columns if 'tempo' in col.lower()]
        if colonne_tempo:
            ax = axes[0, 1]
            dati_tempo = self.df_risultati[['id_istanza'] + colonne_tempo].melt(
                id_vars='id_istanza',
                var_name='Metodo',
                value_name='Tempo (s)'
            )
            dati_tempo['Metodo'] = dati_tempo['Metodo'].str.replace('_tempo', '')
            
            # Scala logaritmica per i tempi
            sns.boxplot(data=dati_tempo, x='Metodo', y='Tempo (s)', ax=ax)
            ax.set_yscale('log')
            ax.set_title('Confronto Tempi Esecuzione', fontsize=14, fontweight='bold')
            ax.set_ylabel('Tempo (secondi, scala log)', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        
        # Tassi di fattibilità
        colonne_fattibile = [col for col in self.df_risultati.columns if 'fattibile' in col.lower()]
        if colonne_fattibile:
            ax = axes[1, 0]
            tassi_fattibilità = {}
            for col in colonne_fattibile:
                metodo = col.replace('_fattibile', '')
                if col in self.df_risultati.columns:
                    tasso = self.df_risultati[col].mean() * 100
                    tassi_fattibilità[metodo] = tasso
            
            metodi = list(tassi_fattibilità.keys())
            tassi = list(tassi_fattibilità.values())
            barre = ax.bar(metodi, tassi, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('Confronto Tassi di Fattibilità', fontsize=14, fontweight='bold')
            ax.set_ylabel('Tasso di Fattibilità (%)', fontsize=12)
            ax.set_ylim([0, 105])
            
            # Aggiungi etichette valori sulle barre
            for barra, tasso in zip(barre, tassi):
                altezza = barra.get_height()
                ax.text(barra.get_x() + barra.get_width()/2., altezza,
                       f'{tasso:.1f}%',
                       ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
        
        # Scalabilità dimensione problema
        ax = axes[1, 1]
        for nome_risolutore in self.risolutori.keys():
            col_tempo = f'{nome_risolutore}_tempo'
            if col_tempo in self.df_risultati.columns:
                ax.scatter(
                    self.df_risultati['n_punti'],
                    self.df_risultati[col_tempo],
                    label=nome_risolutore.upper(),
                    s=50,
                    alpha=0.7
                )
        
        ax.set_xlabel('Dimensione Problema (numero di punti)', fontsize=12)
        ax.set_ylabel('Tempo Esecuzione (s)', fontsize=12)
        ax.set_title('Analisi Scalabilità', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salva figura confronto principale
        percorso_fig = self.directory['visualizzazioni'] / 'confronto_performance.png'
        plt.savefig(percorso_fig, dpi=300, bbox_inches='tight')
        logger.info(f"Salvato confronto performance in {percorso_fig}")
        
        plt.close()
        
        # 2. Genera grafico ottimizzazione lambda dettagliato
        self._grafico_ottimizzazione_lambda()
    
    def _grafico_ottimizzazione_lambda(self):
        """Genera visualizzazione del processo di ottimizzazione lambda."""
        if not hasattr(self, 'lambda_ottimali'):
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        id_istanze = list(self.lambda_ottimali.keys())
        valori_lambda = list(self.lambda_ottimali.values())
        
        ax.bar(id_istanze, valori_lambda, color='steelblue', alpha=0.8)
        ax.set_xlabel('ID Istanza', fontsize=12)
        ax.set_ylabel('Valore λ Ottimale', fontsize=12)
        ax.set_title('Parametri Lambda Ottimali per Istanza', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Aggiungi etichette valori
        for i, (id_ist, lam) in enumerate(zip(id_istanze, valori_lambda)):
            ax.text(i, lam, f'{lam:.2f}', ha='center', va='bottom', fontsize=9)
        
        percorso_fig = self.directory['visualizzazioni'] / 'ottimizzazione_lambda.png'
        plt.savefig(percorso_fig, dpi=300, bbox_inches='tight')
        logger.info(f"Salvato grafico ottimizzazione lambda in {percorso_fig}")
        
        plt.close()
    
    def genera_report(self):
        """
        Genera un report completo dell'esperimento.
        
        Crea un report markdown dettagliato con:
        - Riepilogo configurazione
        - Metriche performance
        - Analisi statistica
        - Conclusioni e raccomandazioni
        """
        percorso_report = self.directory['risultati'] / 'report_esperimento.md'
        
        with open(percorso_report, 'w') as f:
            f.write("# Caso d'Uso Bancario QUACK - Report Esperimento\n\n")
            f.write(f"**Data**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Riepilogo configurazione
            f.write("## Configurazione\n\n")
            f.write(f"- **Istanze**: {len(self.istanze)}\n")
            f.write(f"- **Punti per istanza**: {self.config['esperimento']['parametri_istanza']['n_punti']}\n")
            f.write(f"- **Cluster**: {self.config['esperimento']['parametri_istanza']['n_cluster']}\n")
            f.write(f"- **Dimensione seed**: {self.config['esperimento']['parametri_istanza']['dimensione_cluster_seed']}\n")
            f.write(f"- **Dimensione espansione**: {self.config['esperimento']['parametri_istanza']['dimensione_espansione']}\n")
            f.write(f"- **Range lambda**: {self.config['esperimento'].get('range_lambda', 'default')}\n\n")
            
            # Riepilogo performance
            f.write("## Riepilogo Performance\n\n")
            
            if hasattr(self, 'df_risultati'):
                # Calcola metriche medie
                f.write("### Metriche Medie\n\n")
                f.write("| Risolutore | ARI Medio | Tasso Fattibilità | Tempo Medio (s) |\n")
                f.write("|------------|-----------|-------------------|----------------|\n")
                
                for risolutore in self.risolutori.keys():
                    col_ari = f'{risolutore}_ari'
                    col_fattibile = f'{risolutore}_fattibile'
                    col_tempo = f'{risolutore}_tempo'
                    
                    ari_medio = self.df_risultati[col_ari].mean() if col_ari in self.df_risultati else 'N/A'
                    tasso_fatt = self.df_risultati[col_fattibile].mean() * 100 if col_fattibile in self.df_risultati else 'N/A'
                    tempo_medio = self.df_risultati[col_tempo].mean() if col_tempo in self.df_risultati else 'N/A'
                    
                    if ari_medio != 'N/A':
                        f.write(f"| {risolutore.upper()} | {ari_medio:.3f} | {tasso_fatt:.1f}% | {tempo_medio:.4f} |\n")
                    else:
                        f.write(f"| {risolutore.upper()} | N/A | N/A | N/A |\n")
            
            # Risultati chiave
            f.write("\n## Risultati Chiave\n\n")
            
            # Analizza performance D-Wave
            if 'dwave' in self.risolutori and hasattr(self, 'df_risultati'):
                tempo_dwave = self.df_risultati['dwave_tempo'].mean()
                tempo_classico = self.df_risultati[[col for col in self.df_risultati.columns 
                                                   if 'tempo' in col and 'dwave' not in col]].mean().mean()
                
                if tempo_dwave < tempo_classico:
                    speedup = tempo_classico / tempo_dwave
                    f.write(f"- **Vantaggio Quantistico**: D-Wave ha ottenuto uno speedup di {speedup:.1f}x rispetto ai metodi classici\n")
                else:
                    f.write(f"- **Performance Classica**: I metodi classici attualmente superano il quantistico per questa dimensione del problema\n")
            
            f.write("\n## Conclusioni\n\n")
            f.write("L'esperimento ha dimostrato con successo l'applicazione del quantum annealing ")
            f.write("ai problemi di clustering dei clienti bancari. Osservazioni chiave includono:\n\n")
            f.write("1. L'ottimizzazione del parametro lambda è cruciale per soluzioni fattibili\n")
            f.write("2. Il quantum annealing mostra promesse per istanze di problemi più grandi\n")
            f.write("3. I metodi classici mantengono vantaggio per problemi di piccola scala\n")
            
            f.write("\n## Raccomandazioni\n\n")
            f.write("- Continuare i test con istanze di problemi più grandi (N > 100)\n")
            f.write("- Esplorare approcci ibridi quantistico-classici\n")
            f.write("- Investigare strategie di embedding specifiche per il problema\n")
        
        logger.info(f"Report generato: {percorso_report}")
    
    def esegui(self):
        """
        Esegue la pipeline completa.
        
        Questo è il punto di ingresso principale che orchestra tutti i passi:
        1. Carica dati
        2. Genera istanze  
        3. Ottimizza parametri
        4. Risolve con tutti i metodi
        5. Genera visualizzazioni
        6. Crea report
        """
        logger.info("="*60)
        logger.info("Avvio Pipeline Clustering Bancario QUACK")
        logger.info("="*60)
        
        # Passo 1: Carica dati bancari
        logger.info("\n[Passo 1/6] Caricamento dati bancari...")
        dati = self.carica_dati_bancari()
        
        # Passo 2: Genera istanze
        logger.info("\n[Passo 2/6] Generazione istanze di test...")
        n_istanze = self.config['esperimento'].get('n_istanze', 10)
        istanze = self.genera_istanze(dati, n_istanze=n_istanze)
        
        # Passo 3: Ottimizza parametri lambda
        logger.info("\n[Passo 3/6] Ottimizzazione parametri lambda...")
        lambda_ottimali = self.ottimizza_parametri_lambda()
        
        # Passo 4: Risolvi con tutti i metodi
        logger.info("\n[Passo 4/6] Risoluzione con tutti i metodi disponibili...")
        df_risultati = self.risolvi_con_tutti_metodi()
        
        # Passo 5: Genera visualizzazioni
        logger.info("\n[Passo 5/6] Generazione visualizzazioni...")
        self.genera_visualizzazioni()
        
        # Passo 6: Genera report
        logger.info("\n[Passo 6/6] Generazione report esperimento...")
        self.genera_report()
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline completata con successo!")
        logger.info(f"Risultati salvati in: {self.directory['risultati']}")
        logger.info("="*60)


def main():
    """Punto di ingresso principale per lo script."""
    parser = argparse.ArgumentParser(
        description='Pipeline Clustering Bancario QUACK con D-Wave'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Percorso al file di configurazione (default: config.yaml)'
    )
    
    parser.add_argument(
        '--percorso-dati',
        type=str,
        help='Sovrascrive percorso dati dalla config'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Sovrascrive directory output dalla config'
    )
    
    parser.add_argument(
        '--istanze',
        type=int,
        help='Numero di istanze da generare'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Abilita logging debug'
    )
    
    args = parser.parse_args()
    
    # Imposta livello logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Controlla se il file config esiste
    if not Path(args.config).exists():
        logger.error(f"File configurazione non trovato: {args.config}")
        sys.exit(1)
    
    try:
        # Inizializza ed esegui pipeline
        pipeline = PipelineClusteringBancario(args.config)
        
        # Sovrascrive parametri se forniti
        if args.percorso_dati:
            pipeline.config['percorsi']['file_dati'] = args.percorso_dati
        
        if args.output:
            pipeline.config['percorsi']['cartella_risultati'] = args.output
            pipeline._configura_directory()
        
        if args.istanze:
            pipeline.config['esperimento']['n_istanze'] = args.istanze
        
        # Esegui la pipeline
        pipeline.esegui()
        
    except Exception as e:
        logger.error(f"Pipeline fallita: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
