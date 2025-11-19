"""
Modulo Generatore Istanze Bancarie

Questo modulo crea istanze di test per il caso d'uso bancario del progetto QUACK.
Genera problemi di clustering basati su dati reali di clienti bancari.

Basato sul notebook create_instances_caso_bank.ipynb del progetto.

Autore: Team Progetto QUACK
Data: 2024
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from pathlib import Path

logger = logging.getLogger(__name__)


class GeneratoreIstanzeBancarie:
    """
    Generatore di istanze per il caso d'uso bancario.
    
    Questa classe crea istanze di problemi di clustering basate su
    dati di clienti bancari, includendo preprocessing e generazione
    di matrici di distanza.
    """
    
    def __init__(
        self,
        percorso_dati: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Inizializza il generatore di istanze.
        
        Parametri:
            percorso_dati: Percorso al file CSV con dati bancari
            random_state: Seed per riproducibilità
        """
        self.percorso_dati = percorso_dati
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.dati_processati = None
        
        logger.info("Generatore istanze bancarie inizializzato")
    
    def carica_e_processa_dati(
        self,
        percorso_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carica e preprocessa il dataset bancario.
        
        Basato sul processo del notebook create_instances_caso_bank.ipynb.
        
        Parametri:
            percorso_csv: Percorso al file CSV (usa default se None)
        
        Ritorna:
            DataFrame con dati processati
        """
        if percorso_csv is None:
            percorso_csv = self.percorso_dati
        
        logger.info(f"Caricamento dati da {percorso_csv}")
        
        # Carica dataset
        df = pd.read_csv(percorso_csv)
        
        # Se ci sono coordinate separate, caricale
        try:
            coords_path = percorso_csv.replace('.csv', '_coordinates_and_cluster.csv')
            coords = pd.read_csv(coords_path)
            df['Cluster'] = coords['Cluster'] if 'Cluster' in coords.columns else None
        except:
            logger.info("File coordinate non trovato, procedo senza cluster predefiniti")
        
        # Rimuovi colonne superflue (dal notebook originale)
        colonne_da_rimuovere = [
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
            'AcceptedCmp4', 'AcceptedCmp5', 'Response', 
            'Year_Birth', 'Education', 'Marital_Status', 
            'Kidhome', 'Teenhome', 'Dt_Customer', 
            'Complain', 'Z_CostContact', 'Z_Revenue'
        ]
        
        df = df.drop(columns=colonne_da_rimuovere, errors='ignore')
        
        # Standardizzazione
        colonne_da_scalare = [
            col for col in df.columns 
            if col not in ["ID", "Cluster"]
        ]
        
        if colonne_da_scalare:
            dati_scalati = self.scaler.fit_transform(df[colonne_da_scalare])
            df_scalato = pd.DataFrame(dati_scalati, columns=colonne_da_scalare)
            df_scalato["ID"] = df["ID"].values if "ID" in df else range(len(df))
            
            if "Cluster" in df:
                df_scalato["Cluster"] = df["Cluster"].values
        else:
            df_scalato = df
        
        # Dividi per cluster se presenti
        if "Cluster" in df_scalato:
            self.primo = df_scalato[df_scalato["Cluster"] == "Primo"]
            self.secondo = df_scalato[df_scalato["Cluster"] == "Secondo"]
            logger.info(f"Caricati {len(self.primo)} punti cluster 'Primo', "
                       f"{len(self.secondo)} punti cluster 'Secondo'")
        
        self.dati_processati = df_scalato
        return df_scalato
    
    def genera_configurazioni_istanze(self) -> pd.DataFrame:
        """
        Genera configurazioni per istanze di test.
        
        Basato sulla tabella di configurazione del notebook originale.
        
        Ritorna:
            DataFrame con configurazioni istanze
        """
        # Configurazioni dal notebook (bacini 1000, 2000, 400 punti)
        configurazioni = []
        
        # Template configurazioni per diversi bacini
        templates = [
            # Bacino, proporzione cluster, seed iniziale, punti da aggiungere, percentuale overlap
            (1000, "25_75", 1, 4, 1, 3, "80%", 5),
            (1000, "25_75", 2, 8, 2, 6, "80%", 10),
            (1000, "25_75", 4, 16, 4, 12, "80%", 20),
            (1000, "25_75", 8, 32, 8, 24, "80%", 40),
            (1000, "25_75", 4, 4, 1, 3, "50%", 8),
            (1000, "25_75", 8, 8, 2, 6, "50%", 16),
            (1000, "25_75", 16, 16, 4, 12, "50%", 32),
            (1000, "25_75", 32, 32, 8, 24, "50%", 64),
            (1000, "50_50", 1, 4, 2, 2, "80%", 5),
            (1000, "50_50", 2, 8, 4, 4, "80%", 10),
            (1000, "50_50", 4, 16, 8, 8, "80%", 20),
            (1000, "75_25", 1, 4, 3, 1, "80%", 5),
            (1000, "75_25", 2, 8, 6, 2, "80%", 10),
        ]
        
        for i, config in enumerate(templates):
            configurazioni.append({
                'id': i,
                'bacino': config[0],
                'proporzione': config[1],
                'seed_iniziale': config[2],
                'punti_totali': config[3],
                'punti_cluster1': config[4],
                'punti_cluster2': config[5],
                'overlap': config[6],
                'n_totale': config[7]
            })
        
        return pd.DataFrame(configurazioni)
    
    def genera_istanza(
        self,
        n_punti: int,
        proporzione_cluster: str = "50_50",
        seed_size: int = 10,
        n_espandere: int = 20,
        overlap: float = 0.5
    ) -> Dict:
        """
        Genera singola istanza di clustering.
        
        Parametri:
            n_punti: Numero totale di punti
            proporzione_cluster: Proporzione tra cluster (es. "25_75")
            seed_size: Dimensione cluster iniziale
            n_espandere: Punti da aggiungere
            overlap: Percentuale di overlap tra cluster
        
        Ritorna:
            Dict con istanza generata
        """
        if self.dati_processati is None:
            raise ValueError("Dati non caricati. Chiamare prima carica_e_processa_dati()")
        
        # Campiona punti dal dataset
        if n_punti > len(self.dati_processati):
            logger.warning(f"Richiesti {n_punti} punti ma disponibili solo {len(self.dati_processati)}")
            n_punti = len(self.dati_processati)
        
        # Determina proporzioni cluster
        if proporzione_cluster == "25_75":
            n_cluster1 = int(n_punti * 0.25)
            n_cluster2 = n_punti - n_cluster1
        elif proporzione_cluster == "75_25":
            n_cluster1 = int(n_punti * 0.75)
            n_cluster2 = n_punti - n_cluster1
        else:  # 50_50
            n_cluster1 = n_punti // 2
            n_cluster2 = n_punti - n_cluster1
        
        # Campiona punti per ogni cluster
        if hasattr(self, 'primo') and hasattr(self, 'secondo'):
            # Usa cluster predefiniti se disponibili
            punti_c1 = self.primo.sample(min(n_cluster1, len(self.primo)), 
                                        random_state=self.random_state)
            punti_c2 = self.secondo.sample(min(n_cluster2, len(self.secondo)), 
                                         random_state=self.random_state)
            dati_campionati = pd.concat([punti_c1, punti_c2])
        else:
            # Campiona casualmente
            dati_campionati = self.dati_processati.sample(n_punti, 
                                                         random_state=self.random_state)
        
        # Rimuovi colonne non numeriche per matrice distanze
        colonne_numeriche = dati_campionati.select_dtypes(include=[np.number]).columns
        valori_numerici = dati_campionati[colonne_numeriche].values
        
        # Calcola matrice distanze
        matrice_distanze = pairwise_distances(valori_numerici, metric='euclidean')
        
        # Seleziona seed cluster (primi seed_size punti del primo cluster)
        indici_seed = np.random.choice(n_cluster1, size=min(seed_size, n_cluster1), 
                                      replace=False)
        
        # Determina punti candidati per espansione
        tutti_indici = set(range(len(dati_campionati)))
        indici_candidati = np.array(list(tutti_indici - set(indici_seed)))
        
        # Crea ground truth per valutazione
        y_true = np.zeros(len(dati_campionati))
        y_true[n_cluster1:] = 1  # Assegna secondo cluster
        
        istanza = {
            'matrice_distanze': matrice_distanze,
            'dati': dati_campionati,
            'cluster_seed': indici_seed,
            'n_espandere': min(n_espandere, len(indici_candidati)),
            'punti_candidati': indici_candidati,
            'y_true': y_true,
            'info': {
                'n_punti': len(dati_campionati),
                'n_cluster': 2,
                'proporzione': proporzione_cluster,
                'overlap': overlap,
                'seed_size': len(indici_seed)
            }
        }
        
        return istanza
    
    def genera_batch_istanze(
        self,
        configurazioni: pd.DataFrame,
        percorso_output: Optional[str] = None
    ) -> List[Dict]:
        """
        Genera batch di istanze basate su configurazioni.
        
        Parametri:
            configurazioni: DataFrame con configurazioni
            percorso_output: Directory per salvare istanze
        
        Ritorna:
            Lista di istanze generate
        """
        istanze = []
        
        if percorso_output:
            Path(percorso_output).mkdir(parents=True, exist_ok=True)
        
        for idx, row in configurazioni.iterrows():
            logger.info(f"Generazione istanza {idx}: bacino={row['bacino']}, "
                       f"proporzione={row['proporzione']}")
            
            try:
                istanza = self.genera_istanza(
                    n_punti=row['bacino'],
                    proporzione_cluster=row['proporzione'],
                    seed_size=row['seed_iniziale'],
                    n_espandere=row['punti_totali'] - row['seed_iniziale'],
                    overlap=float(row['overlap'].rstrip('%')) / 100
                )
                
                istanza['id'] = idx
                istanze.append(istanza)
                
                # Salva istanza se richiesto
                if percorso_output:
                    nome_file = Path(percorso_output) / f"istanza_{idx}.pkl"
                    self.salva_istanza(istanza, str(nome_file))
                    
            except Exception as e:
                logger.error(f"Errore generazione istanza {idx}: {e}")
        
        logger.info(f"Generate {len(istanze)} istanze")
        return istanze
    
    def salva_istanza(self, istanza: Dict, percorso_file: str):
        """
        Salva istanza su file pickle.
        
        Parametri:
            istanza: Dizionario istanza
            percorso_file: Percorso file output
        """
        # Rimuovi DataFrame per compatibilità pickle
        istanza_da_salvare = istanza.copy()
        if 'dati' in istanza_da_salvare:
            # Converti DataFrame in dict o array se necessario
            istanza_da_salvare['dati'] = istanza_da_salvare['dati'].to_dict()
        
        with open(percorso_file, 'wb') as f:
            pickle.dump(istanza_da_salvare, f)
        
        logger.info(f"Istanza salvata in '{percorso_file}'")
    
    @staticmethod
    def carica_istanza(percorso_file: str) -> Dict:
        """
        Carica istanza da file pickle.
        
        Parametri:
            percorso_file: Percorso file input
        
        Ritorna:
            Dict istanza caricata
        """
        with open(percorso_file, 'rb') as f:
            istanza = pickle.load(f)
        
        # Riconverti dati in DataFrame se necessario
        if 'dati' in istanza and isinstance(istanza['dati'], dict):
            istanza['dati'] = pd.DataFrame(istanza['dati'])
        
        logger.info("Istanza caricata con successo")
        return istanza
    
    def genera_istanze_test_standard(
        self,
        n_istanze: int = 10,
        percorso_output: Optional[str] = None
    ) -> List[Dict]:
        """
        Genera set standard di istanze per test.
        
        Parametri:
            n_istanze: Numero di istanze da generare
            percorso_output: Directory output
        
        Ritorna:
            Lista istanze generate
        """
        # Configurazioni di default per test
        configurazioni_test = []
        
        for i in range(n_istanze):
            # Varia parametri per diversità
            n_punti = np.random.choice([50, 100, 200, 400])
            proporzione = np.random.choice(["25_75", "50_50", "75_25"])
            seed_size = max(5, n_punti // 20)
            n_espandere = max(10, n_punti // 10)
            
            configurazioni_test.append({
                'id': i,
                'bacino': n_punti,
                'proporzione': proporzione,
                'seed_iniziale': seed_size,
                'punti_totali': seed_size + n_espandere,
                'punti_cluster1': n_punti // 2,
                'punti_cluster2': n_punti // 2,
                'overlap': "50%",
                'n_totale': n_punti
            })
        
        df_config = pd.DataFrame(configurazioni_test)
        
        return self.genera_batch_istanze(df_config, percorso_output)
