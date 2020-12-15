# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
import numpy as np

# Création d'un client pour les outils de visualisation
from dask.distributed import Client, progress
from dask_yarn import YarnCluster

cluster = YarnCluster()
client = Client(cluster)

cluster.adapt() 

client

# %% [markdown]
# # Chargement des dataset
# %% [markdown]
# ## YouGov - Wearing Mask in public

# %%
start = datetime.now()


##Chargement dataset
df = dd.read_csv(
    "gs://dask-vs-koalas/wearing_face_mask_public.csv",
    sep=";"
)

##Transformation du dataset = 1 ligne par date/pays
format = '%Y-%m-%d %H:%M:%S'
df['DateTime'] = dd.to_datetime(df['DateTime'], format=format)
df['DateTime'] = df['DateTime'].dt.normalize()

##### 1er changement: sort_index et sort_values n'existe pas, d'ailleurs Dask ne supporte pas le tri sur de multiple colonnes
##### Dans ce cas le sort_values n'était pas nécessaire et le groupby non plus
# df = df.sort_values('DateTime').groupby(df['DateTime']).max()
df = df.set_index('DateTime')
##### 2e changement: pad n'existe pas sur un resample (Dask Resampler) seules les fonctions de downsampling sont implémentées
# df = df.resample('1D').pad()
wearing_mask_in_public_data = df.resample('1D').last()

wearing_mask_in_public_data = wearing_mask_in_public_data.fillna(0)
wearing_mask_in_public_data = wearing_mask_in_public_data.reset_index().melt(
                                id_vars=['DateTime'], 
                                var_name='country', 
                                value_name='percent_wearing_mask')


print(f"Le dataset contient {len(df)} enregistrements")

##### 3e changement avec Dask : la doc indique que df.sample(5) n'est pas accepté, le paramère "n"
##### ne doit pas être utlisé. Il faut utiliser "frac"
##### De plus le "print" d'un Dataframe Dask n'affichera pas toutes les valeurs mais les types
##### Enfin pour profiter du laziness on n'affichera pas avant la fin
print("Sample dataset final:")
print(wearing_mask_in_public_data.head(5))

stop = datetime.now()

print("Temps de chargement et tranformation petit dataset : ", (stop-start).microseconds/1000, "ms")

# %% [markdown]
# ## Google - Covid 19 Open Data

# %%
start = datetime.now()

#Chargement dataset

##### 4e changement avec Dask : inférence de type
# On va spécifier les types en erreur
types = {"locality_code": str, "locality_name": str, "subregion1_code": str, "subregion1_name": str, "subregion2_name": str, "subregion2_code": str}
covid19_opendata = dd.read_csv(
    "gs://dask-vs-koalas/main.csv",
    keep_default_na=False,
    na_values=[""],
    dtype=types,
    sample=10000000)



# Jointure entre open data covid 19 et yougo
#covid19_opendata['date'] = pd.to_datetime(covid19_opendata['date'], format=format)
covid19_opendata['date'] = covid19_opendata['date'].astype('M8[D]')

covid19_merge1 = covid19_opendata.merge(wearing_mask_in_public_data, 
                                      left_on = ['country_name','date'],
                                      right_on = ['country','DateTime'], how = 'left')


remove_cols = ['key', 'country','aggregation_level','locality_code', 'wikidata', 'datacommons', 'country_code', 'subregion1_code', 'subregion1_name', 'subregion2_code', 'subregion2_name', 'locality_name', '3166-1-alpha-2', '3166-1-alpha-3', 'DateTime']

covid19_merge1 = covid19_merge1.drop(remove_cols, axis=1)
covid19_merge1 = covid19_merge1.fillna(0)

#prepared_data =  covid19_merge1.copy()
#### 5e changement : exécuté avec des "workers" qui acceptent 8Go max, il y aura une consommation excessive de la mémoire
#### On avait 146 partitions, on diminue de moitié la taille des partitions actuelles
print("Partitions avant: ", covid19_merge1.npartitions)
covid19_merge1 = covid19_merge1.repartition(npartitions=covid19_merge1.npartitions * 2)
print("Partitions après: ", covid19_merge1.npartitions)

#print("covid19_merge1 partition :",covid19_merge1.npartitions)
prepared_data = client.persist(covid19_merge1)


## Encode Pays
from dask_ml.preprocessing import LabelEncoder
encoded_countries = LabelEncoder().fit_transform(prepared_data.country_name)
prepared_data['country_name'] = encoded_countries

## Encode Date
dates = prepared_data.date.apply(lambda x: x.strftime('%Y%m%d'))
encoded_dates = LabelEncoder().fit_transform(dates)
prepared_data['date'] = encoded_dates


print(f"Le dataset contient {len(prepared_data)} enregistrements")
print("Sample dataset final:")
print(prepared_data.head(5))

stop = datetime.now()


print("Temps de chargement et tranformation grand dataset : ", (stop-start).seconds, "s")

# %% [markdown]
# ## Entrainement et inférence

# %%
#### 6e changement : Utilisation d'un backend spécifique pour dask
import joblib

start = datetime.now()

# Split Train/Testmain
from dask_ml.model_selection import train_test_split
X = prepared_data.loc[:, prepared_data.columns != 'new_confirmed']    

#### 7e changement : pour la création des labels Y, conversion de Series en Dask array
y = prepared_data['new_confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Scale des valeurs
from dask_ml.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


regr = MLPRegressor(max_iter=10, hidden_layer_sizes=(100, 50, 25, 10, 5), verbose=True)

#### 8e changement : Parallélisme pour l'entrainement et la prédiction
with joblib.parallel_backend('dask'):
    regr.fit(X_train, y_train)

# Prédiction et Score
with joblib.parallel_backend('dask'):
    score = regr.score(X_test, y_test)

stop = datetime.now()

print("Temps préparation et inférence (ML) : ", (stop-start).seconds, "s")
print(f"model score: {score}")

# %% [markdown]
# Seul le training et le scale est parallélisé par Dask car le MLPRegressor n'a pas d'implémentation "Dask"
# %% [markdown]
# ## Entrainement et inférence avec pipeline

# %%
#### 6e changement : Utilisation d'un backend spécifique pour dask
import joblib

start = datetime.now()

# Split Train/Test
from dask_ml.model_selection import train_test_split
X = prepared_data.loc[:, prepared_data.columns != 'new_confirmed']

#### 7e changement : pour la création des labels Y, conversion de Series en Dask array
y = prepared_data['new_confirmed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.pipeline import Pipeline

# Scale des valeurs
from dask_ml.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

scaler = StandardScaler()
regr = MLPRegressor(max_iter=500, hidden_layer_sizes=(100,25,5))

pipeline = Pipeline([('scaler', scaler), ('regressor', regr)])

# Exécution du pipeline
with joblib.parallel_backend('dask'):
    pipeline.fit(X_train, y_train)

# Prédiction et Score
with joblib.parallel_backend('dask'):
    score = pipeline.score(X_test, y_test)

stop = datetime.now()

print("Temps préparation et inférence (ML) : ", (stop-start).seconds, "s")
print(f"model score: {score}")


