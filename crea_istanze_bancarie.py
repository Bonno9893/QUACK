"""
Generazione delle istanze bancarie per l’Algoritmo 1 (progetto QUACK).

Questo script parte dal dataset pubblico "New Marketing Campaign"
(disponibile su Kaggle) già preprocessato e costruisce le istanze
utilizzate nel caso d’uso bancario del progetto QUACK. In particolare,
seleziona i sottoinsiemi di punti (seed e candidati), calcola le
matrici di distanza e salva le istanze in formato intermedio (.pkl),
equivalente alle versioni testuali (.txt) presenti nella cartella
delle istanze di questa repository.

Il file è incluso principalmente a scopo di consultazione, per
documentare il processo di generazione delle istanze originali.
Per replicare gli esperimenti descritti nel paper è sufficiente
utilizzare le istanze .txt e i valori di lambda forniti nel file
lambda.csv.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import numpy as np
from scipy.spatial.distance import pdist,squareform
import random
import pickle


df = pd.read_csv("best_1000_points_full_attributes.csv")
coords = pd.read_csv('best_1000_points_coordinates_and_cluster.csv')
distance_mat = pd.read_csv('best_1000_points_distance_matrix.csv', header=None)

# 3. Rimozione di colonne superflue
columns_to_remove = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome','Dt_Customer', 'Complain', 'Z_CostContact', 'Z_Revenue']
df = df.drop(columns=columns_to_remove, errors='ignore')

# 5. Standardizzazione
columns_to_scale = [col for col in df.columns if col != "ID" and col != "Cluster"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[columns_to_scale])
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
scaled_df["ID"] = df["ID"].values
scaled_df["Cluster"] = df["Cluster"].values
primo = scaled_df[scaled_df["Cluster"] == "Primo"]
secondo = scaled_df[scaled_df["Cluster"] == "Secondo"]

data_instance_1000 = pd.read_csv(io.StringIO('''
0,Bacino 1000,25 E 75 ,1,4,1,3,80%,5
1,Bacino 1000,25 E 75,2,8,2,6,80%,10
2,Bacino 1000,25 E 75,4,16,4,12,80%,20
3,Bacino 1000,25 E 75,8,32,8,24,80%,40
4,Bacino 1000,25 E 75,4,4,1,3,50%,8
5,Bacino 1000,25 E 75,8,8,2,6,50%,16
6,Bacino 1000,25 E 75,16,16,4,12,50%,32
7,Bacino 1000,25 E 75,32,32,8,24,50%,64
8,Bacino 1000,25 E 75,6,4,1,3,40%,10
9,Bacino 1000,25 E 75,12,8,2,6,40%,20
10,Bacino 1000,25 E 75,24,16,4,12,40%,40
11,Bacino 1000,25 E 75,48,32,8,24,40%,80
12,Bacino 1000,25 E 75,16,4,1,3,20%,20
13,Bacino 1000,25 E 75,32,8,2,6,20%,40
14,Bacino 1000,25 E 75,64,16,4,12,20%,80
15,Bacino 1000,25 E 75,128,32,8,24,20%,160
16,Bacino 1000,50 E 50,1,4,2,2,80%,5
17,Bacino 1000,50 E 50,2,8,4,4,80%,10
18,Bacino 1000,50 E 50,4,16,8,8,80%,20
19,Bacino 1000,50 E 50,8,32,16,16,80%,40
20,Bacino 1000,50 E 50,4,4,2,2,50%,8
21,Bacino 1000,50 E 50,8,8,4,4,50%,16
22,Bacino 1000,50 E 50,16,16,8,8,50%,32
23,Bacino 1000,50 E 50,32,32,16,16,50%,64
24,Bacino 1000,50 E 50,6,4,2,2,40%,10
25,Bacino 1000,50 E 50,12,8,4,4,40%,20
26,Bacino 1000,50 E 50,24,16,8,8,40%,40
27,Bacino 1000,50 E 50,48,32,16,16,40%,80
28,Bacino 1000,50 E 50,16,4,2,2,20%,20
29,Bacino 1000,50 E 50,32,8,4,4,20%,40
30,Bacino 1000,50 E 50,64,16,8,8,20%,80
31,Bacino 1000,50 E 50,128,32,16,16,20%,160
32,Bacino 1000,75 E 25,1,4,3,1,80%,5
33,Bacino 1000,75 E 25,2,8,6,2,80%,10
34,Bacino 1000,75 E 25,4,16,12,4,80%,20
35,Bacino 1000,75 E 25,8,32,24,8,80%,40
36,Bacino 1000,75 E 25,4,4,3,1,50%,8
37,Bacino 1000,75 E 25,8,8,6,2,50%,16
38,Bacino 1000,75 E 25,16,16,12,4,50%,32
39,Bacino 1000,75 E 25,32,32,24,8,50%,64
40,Bacino 1000,75 E 25,6,4,3,1,40%,10
41,Bacino 1000,75 E 25,12,8,6,2,40%,20
42,Bacino 1000,75 E 25,24,16,12,4,40%,40
43,Bacino 1000,75 E 25,48,32,24,8,40%,80
44,Bacino 1000,75 E 25,16,4,3,1,20%,20
45,Bacino 1000,75 E 25,32,8,6,2,20%,40
46,Bacino 1000,75 E 25,64,16,12,4,20%,80
47,Bacino 1000,75 E 25,128,32,24,8,20%,160
'''), header=None)


data_instance_2000 = pd.read_csv(io.StringIO('''
48,Bacino 2000,25 E 75 ,1,4,1,3,80%,5
49,Bacino 2000,25 E 75,2,8,2,6,80%,10
50,Bacino 2000,25 E 75,4,16,4,12,80%,20
51,Bacino 2000,25 E 75,8,32,8,24,80%,40
52,Bacino 2000,25 E 75,4,4,1,3,50%,8
53,Bacino 2000,25 E 75,8,8,2,6,50%,16
54,Bacino 2000,25 E 75,16,16,4,12,50%,32
55,Bacino 2000,25 E 75,32,32,8,24,50%,64
56,Bacino 2000,25 E 75,6,4,1,3,40%,10
57,Bacino 2000,25 E 75,12,8,2,6,40%,20
58,Bacino 2000,25 E 75,24,16,4,12,40%,40
59,Bacino 2000,25 E 75,48,32,8,24,40%,80
60,Bacino 2000,25 E 75,16,4,1,3,20%,20
61,Bacino 2000,25 E 75,32,8,2,6,20%,40
62,Bacino 2000,25 E 75,64,16,4,12,20%,80
63,Bacino 2000,25 E 75,128,32,8,24,20%,160
64,Bacino 2000,50 E 50,1,4,2,2,80%,5
65,Bacino 2000,50 E 50,2,8,4,4,80%,10
66,Bacino 2000,50 E 50,4,16,8,8,80%,20
67,Bacino 2000,50 E 50,8,32,16,16,80%,40
68,Bacino 2000,50 E 50,4,4,2,2,50%,8
69,Bacino 2000,50 E 50,8,8,4,4,50%,16
70,Bacino 2000,50 E 50,16,16,8,8,50%,32
71,Bacino 2000,50 E 50,32,32,16,16,50%,64
72,Bacino 2000,50 E 50,6,4,2,2,40%,10
73,Bacino 2000,50 E 50,12,8,4,4,40%,20
74,Bacino 2000,50 E 50,24,16,8,8,40%,40
75,Bacino 2000,50 E 50,48,32,16,16,40%,80
76,Bacino 2000,50 E 50,16,4,2,2,20%,20
77,Bacino 2000,50 E 50,32,8,4,4,20%,40
78,Bacino 2000,50 E 50,64,16,8,8,20%,80
79,Bacino 2000,50 E 50,128,32,16,16,20%,160
80,Bacino 2000,75 E 25,1,4,3,1,80%,5
81,Bacino 2000,75 E 25,2,8,6,2,80%,10
82,Bacino 2000,75 E 25,4,16,12,4,80%,20
83,Bacino 2000,75 E 25,8,32,24,8,80%,40
84,Bacino 2000,75 E 25,4,4,3,1,50%,8
85,Bacino 2000,75 E 25,8,8,6,2,50%,16
86,Bacino 2000,75 E 25,16,16,12,4,50%,32
87,Bacino 2000,75 E 25,32,32,24,8,50%,64
88,Bacino 2000,75 E 25,6,4,3,1,40%,10
89,Bacino 2000,75 E 25,12,8,6,2,40%,20
90,Bacino 2000,75 E 25,24,16,12,4,40%,40
91,Bacino 2000,75 E 25,48,32,24,8,40%,80
92,Bacino 2000,75 E 25,16,4,3,1,20%,20
93,Bacino 2000,75 E 25,32,8,6,2,20%,40
94,Bacino 2000,75 E 25,64,16,12,4,20%,80
95,Bacino 2000,75 E 25,128,32,24,8,20%,160
'''), header=None)


data_instance_400 = pd.read_csv(io.StringIO('''
96,Bacino 400,25 E 75 ,1,4,1,3,80%,5
97,Bacino 400,25 E 75,2,8,2,6,80%,10
98,Bacino 400,25 E 75,4,16,4,12,80%,20
99,Bacino 400,25 E 75,8,32,8,24,80%,40
100,Bacino 400,25 E 75,4,4,1,3,50%,8
101,Bacino 400,25 E 75,8,8,2,6,50%,16
102,Bacino 400,25 E 75,16,16,4,12,50%,32
103,Bacino 400,25 E 75,32,32,8,24,50%,64
104,Bacino 400,25 E 75,6,4,1,3,40%,10
105,Bacino 400,25 E 75,12,8,2,6,40%,20
106,Bacino 400,25 E 75,24,16,4,12,40%,40
107,Bacino 400,25 E 75,48,32,8,24,40%,80
108,Bacino 400,25 E 75,16,4,1,3,20%,20
109,Bacino 400,25 E 75,32,8,2,6,20%,40
110,Bacino 400,25 E 75,64,16,4,12,20%,80
111,Bacino 400,25 E 75,128,32,8,24,20%,160
112,Bacino 400,50 E 50,1,4,2,2,80%,5
113,Bacino 400,50 E 50,2,8,4,4,80%,10
114,Bacino 400,50 E 50,4,16,8,8,80%,20
115,Bacino 400,50 E 50,8,32,16,16,80%,40
116,Bacino 400,50 E 50,4,4,2,2,50%,8
117,Bacino 400,50 E 50,8,8,4,4,50%,16
118,Bacino 400,50 E 50,16,16,8,8,50%,32
119,Bacino 400,50 E 50,32,32,16,16,50%,64
120,Bacino 400,50 E 50,6,4,2,2,40%,10
121,Bacino 400,50 E 50,12,8,4,4,40%,20
122,Bacino 400,50 E 50,24,16,8,8,40%,40
123,Bacino 400,50 E 50,48,32,16,16,40%,80
124,Bacino 400,50 E 50,16,4,2,2,20%,20
125,Bacino 400,50 E 50,32,8,4,4,20%,40
126,Bacino 400,50 E 50,64,16,8,8,20%,80
127,Bacino 400,50 E 50,128,32,16,16,20%,160
128,Bacino 400,75 E 25,1,4,3,1,80%,5
129,Bacino 400,75 E 25,2,8,6,2,80%,10
130,Bacino 400,75 E 25,4,16,12,4,80%,20
131,Bacino 400,75 E 25,8,32,24,8,80%,40
132,Bacino 400,75 E 25,4,4,3,1,50%,8
133,Bacino 400,75 E 25,8,8,6,2,50%,16
134,Bacino 400,75 E 25,16,16,12,4,50%,32
135,Bacino 400,75 E 25,32,32,24,8,50%,64
136,Bacino 400,75 E 25,6,4,3,1,40%,10
137,Bacino 400,75 E 25,12,8,6,2,40%,20
138,Bacino 400,75 E 25,24,16,12,4,40%,40
139,Bacino 400,75 E 25,48,32,24,8,40%,80
140,Bacino 400,75 E 25,16,4,3,1,20%,20
141,Bacino 400,75 E 25,32,8,6,2,20%,40
142,Bacino 400,75 E 25,64,16,12,4,20%,80
143,Bacino 400,75 E 25,128,32,24,8,20%,160
'''), header=None)


data_instance_2000_big = pd.read_csv(io.StringIO('''
144,Bacino 2000,25 E 75,16,64,16,48,80%,80
145,Bacino 2000,25 E 75,64,64,16,48,50%,128
146,Bacino 2000,25 E 75,96,64,16,48,40%,160
147,Bacino 2000,25 E 75,256,64,16,48,20%,320
148,Bacino 2000,50 E 50,16,64,32,32,80%,80
149,Bacino 2000,50 E 50,64,64,32,32,50%,128
150,Bacino 2000,50 E 50,96,64,32,32,40%,160
151,Bacino 2000,50 E 50,256,64,32,32,20%,320
152,Bacino 2000,75 E 25,16,64,48,16,80%,80
153,Bacino 2000,75 E 25,64,64,48,16,50%,128
154,Bacino 2000,75 E 25,96,64,48,16,40%,160
155,Bacino 2000,75 E 25,256,64,48,16,20%,320
156,Bacino 2000,25 E 75,32,128,32,96,80%,160
157,Bacino 2000,25 E 75,128,128,32,96,50%,256
158,Bacino 2000,25 E 75,192,128,32,96,40%,320
159,Bacino 2000,25 E 75,512,128,32,96,20%,640
160,Bacino 2000,50 E 50,32,128,64,64,80%,160
161,Bacino 2000,50 E 50,128,128,64,64,50%,256
162,Bacino 2000,50 E 50,192,128,64,64,40%,320
163,Bacino 2000,50 E 50,512,128,64,64,20%,640
164,Bacino 2000,75 E 25,32,128,96,32,80%,160
165,Bacino 2000,75 E 25,128,128,96,32,50%,256
166,Bacino 2000,75 E 25,192,128,96,32,40%,320
167,Bacino 2000,75 E 25,512,128,96,32,20%,640
'''), header=None)


data_instance_400_big  = pd.read_csv(io.StringIO('''
168,Bacino 400,25 E 75,20,80,20,60,80%,100
169,Bacino 400,25 E 75,80,80,20,60,50%,160
170,Bacino 400,25 E 75,120,80,20,60,40%,200
171,Bacino 400,25 E 75,320,80,20,60,20%,400
172,Bacino 400,50 E 50,20,80,40,40,80%,100
173,Bacino 400,50 E 50,80,80,40,40,50%,160
174,Bacino 400,50 E 50,120,80,40,40,40%,200
175,Bacino 400,50 E 50,320,80,40,40,20%,400
176,Bacino 400,75 E 25,20,80,60,20,80%,100
177,Bacino 400,75 E 25,80,80,60,20,50%,160
178,Bacino 400,75 E 25,120,80,60,20,40%,200
179,Bacino 400,75 E 25,320,80,60,20,20%,400
180,Bacino 400,25 E 75,16,64,16,48,80%,80
181,Bacino 400,25 E 75,64,64,16,48,50%,128
182,Bacino 400,25 E 75,96,64,16,48,40%,160
183,Bacino 400,25 E 75,256,64,16,48,20%,320
184,Bacino 400,50 E 50,16,64,32,32,80%,80
185,Bacino 400,50 E 50,64,64,32,32,50%,128
186,Bacino 400,50 E 50,96,64,32,32,40%,160
187,Bacino 400,50 E 50,256,64,32,32,20%,320
188,Bacino 400,75 E 25,16,64,48,16,80%,80
189,Bacino 400,75 E 25,64,64,48,16,50%,128
190,Bacino 400,75 E 25,96,64,48,16,40%,160
191,Bacino 400,75 E 25,256,64,48,16,20%,320
'''), header=None)

# prompt: crea una fuzione che per ogni riga del dataframe data_instance prende un numero di punti pari al valore dell'ultima colonna e ne sceglie a caso {valore della colonna 7} dal dataset primo, {valore della colonna 6} dal dataset secondo ed i rimanenti di nuovo dal dataset primo (escludendo quelli già scielti), dati questi punti scelti così crea una matrice delle distanze di tutti i punti utilizzando la funzione pdist

def create_distance_matrix(data_instance, primo, secondo, coords, distance_mat):
  results = []
  for index, row in data_instance.iterrows():
    num_points = row.iloc[-1]  # Numero di punti dall'ultima colonna
    from_primo = row.iloc[6] # numero di punti dal primo dataset
    from_secondo = row.iloc[5] # numero di punti dal secondo dataset
    from_I0 = row.iloc[3] # numero di punti in I0

    #print(from_I0)

    from_primo_tot = from_primo + (num_points - from_secondo - from_primo)
    # Seleziona punti casuali
    selected_points = []

    # Punti da primo dataset
    primo_sample = primo.sample(n=from_primo_tot, replace=False)
    selected_points.extend(primo_sample.values.tolist())


    # Punti da secondo dataset
    secondo_sample = secondo.sample(n=from_secondo, replace=False)
    selected_points.extend(secondo_sample.values.tolist())

    lista_indici = list(primo_sample.index) + list(secondo_sample.index)

    #print(lista_indici)

    coords_selected = coords.loc[lista_indici]
    #print(coords_selected)

    distance_matrix = distance_mat.loc[lista_indici,lista_indici]

    #print(distance_matrix)

    index_i0 = random.sample(list(primo_sample.index),from_I0)
    n_points_to_add = from_primo
    results.append({'dm':distance_matrix,'idx_i0':index_i0,'n_pints_to_add':n_points_to_add, 'coords_selected':coords_selected})


  return results

print(np.__version__)

instances = create_distance_matrix(data_instance_400_big,primo,secondo,coords, distance_mat)

def salva_istanza(instance, nome_file):
    with open(nome_file, 'wb') as file:
        pickle.dump(instance, file)
    print(f"Istanza salvata in '{nome_file}'")

def carica_istanza(nome_file):
    # Carica il modello
    with open(nome_file, 'rb') as file:
        models = pickle.load(file)
    print("Istanza caricata con successo.")

    return models

instances[0]

count = 168
for elem in instances:
  salva_istanza(elem,f'istanza_{count}.pkl')
  count += 1

ist = carica_istanza('istanza_0.pkl')

ist['dm']

df_attributes = pd.read_csv('best_1000_points_full_attributes.csv')

df_attributes.loc[ist['dm'].index][df_attributes.loc[ist['dm'].index]['Cluster']=='Primo'].describe()

df_attributes.loc[ist['dm'].index][df_attributes.loc[ist['dm'].index]['Cluster']=='Secondo'].describe()
