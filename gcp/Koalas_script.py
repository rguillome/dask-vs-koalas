# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession
from datetime import datetime

ks.set_option('compute.default_index_type', 'distributed')

# %% [markdown]
# ## YouGov - Wearing Mask in public

# %%
start = datetime.now()


##Chargement dataset
df = ks.read_csv(
    "gs://dask-vs-koalas/wearing_face_mask_public.csv",
    sep=";"
)

##Transformation du dataset = 1 ligne par date/pays
format = '%Y-%m-%d %H:%M:%S'
df['DateTime'] = ks.to_datetime(df['DateTime'], format=format)
df['DateTime'] = df['DateTime'].dt.normalize()


#### 1er changement : autoriser les opérations sur 2 dataframes différents (ks.set_option('compute.ops_on_diff_frames', True) 
#### ou faire un groupby sur la colonne (comportement légèrement différent de pandas car la colonne de group_by devient un index et disparait de la projection)
# df = df.sort_values('DateTime').groupby(df['DateTime']).max()
df = df.sort_values('DateTime').groupby(['DateTime'], as_index=False).max()
# df = df.set_index(pd.DatetimeIndex(df['DateTime'])).drop(['DateTime'], axis=1)
df = df.set_index('DateTime')

#### 2e changement : The method `pd.DataFrame.resample()` is not implemented yet. (en cours d'étude : https://github.com/databricks/koalas/issues/1562)
#### on est obligé de partir sur Spark directement dans ce cas ou alors de passer par pandas ...
df = df.to_pandas()
wearing_mask_in_public_data = df.resample('1D').pad()

#### Retours au dataframe Koalas
wearing_mask_in_public_data = ks.from_pandas(wearing_mask_in_public_data)
wearing_mask_in_public_data = wearing_mask_in_public_data.fillna(0)
wearing_mask_in_public_data = wearing_mask_in_public_data.reset_index().melt(
                                id_vars=['DateTime'], 
                                var_name='country', 
                                value_name='percent_wearing_mask')


stop = datetime.now()

print("Temps de chargement et tranformation petit dataset : ", (stop-start).microseconds/1000, "ms")
print(f"Le dataset contient {len(df)} enregistrements")

print("Sample dataset final:")
##### 3e changement : Function sample currently does not support specifying exact number of items to return. Use frac instead.
print(wearing_mask_in_public_data.head(5))

# %% [markdown]
# ## Google - Covid 19 Open Data

# %%
start = datetime.now()

#Chargement dataset
covid19_opendata = ks.read_csv(
    "gs://dask-vs-koalas/main.csv",
    keep_default_na=False,
    na_values=[""])



# Jointure entre open data covid 19 et yougo
format = '%Y-%m-%d %H:%M:%S'
##### 4e changement : la transformation de datetime en date ne se fait pas directement
covid19_opendata['date'] = ks.to_datetime(covid19_opendata['date'], format=format)
covid19_opendata['date'] = covid19_opendata['date'].dt.normalize()


covid19_merge1 = covid19_opendata.merge(wearing_mask_in_public_data, 
                                      left_on = ['country_name','date'],
                                      right_on = ['country','DateTime'], how = 'left')


remove_cols = ['key', 'country','aggregation_level','locality_code', 'wikidata', 'datacommons', 'country_code', 'subregion1_code', 'subregion1_name', 'subregion2_code', 'subregion2_name', 'locality_name', '3166-1-alpha-2', '3166-1-alpha-3', 'DateTime']

covid19_merge1 = covid19_merge1.drop(remove_cols, axis=1)

prepared_data =  covid19_merge1.copy()

#### 5e changement, Les fonctions de préprocessing de scikit learn ne sont pas accessibles avec les dataframes Koalas 
#### Si on repasse en dataframe pandas, tout est remonté au driver donc on va plutôt utiliser get_dummies
prepared_data = ks.get_dummies(prepared_data, columns=['country_name', 'date'])


prepared_data = prepared_data.fillna(0)

print(f"Le dataset contient {len(prepared_data)} enregistrements")

print("Sample dataset final:")
print(prepared_data.head(5))

stop = datetime.now()

print("Temps de chargement et tranformation grand dataset : ", (stop-start).seconds, "s")

# %% [markdown]
# ## Entrainement et inférence

# %%
start = datetime.now()

prepared_data = prepared_data.spark.persist()
##### 6e changement : Pour utiliser scikit learn avec koalas, il faut utiliser Mlflow
##### Mais l'entrainement restera sur des dataframe pandas, seule la prédiction peut être faite avec koalas
prepared_data = prepared_data.to_pandas()

#### On prépare donc l'environnement
from mlflow.tracking import MlflowClient, set_tracking_uri
import mlflow.sklearn

from tempfile import mkdtemp
d = mkdtemp("koalas_mlflow")
set_tracking_uri("file:%s"%d)
client = MlflowClient()
exp = mlflow.create_experiment("my_experiment")
mlflow.set_experiment("my_experiment")

# Split Train/Test
from sklearn.model_selection import train_test_split
X = prepared_data.loc[:, prepared_data.columns != 'new_confirmed']
y = prepared_data['new_confirmed'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Scale des valeurs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Entraintement MLP
from sklearn.neural_network import MLPRegressor

with mlflow.start_run():
    regr = MLPRegressor(max_iter=10, hidden_layer_sizes=(100, 50, 25, 10, 5), verbose=True)
   
    regr.fit(X_train, y_train)

    mlflow.sklearn.log_model(regr, "model")


#### Notre modèle est entrainé, on peut donc l'utiliser sur des datafames Koalas
from databricks.koalas.mlflow import load_model
run_info = client.list_run_infos(exp)[-1]

model = load_model("runs:/{run_id}/model".format(run_id=run_info.run_uuid))

# Prédiction et Score
df = ks.DataFrame(X_test)
df["prediction"] = model.predict(df)

stop = datetime.now()

print("Temps préparation et inférence (ML) : ", (stop-start).seconds, "s")


# %%
##### 7e changement : Il faut donc recalculer le score nous même

from databricks.koalas.config import set_option, reset_option

set_option("compute.ops_on_diff_frames", True)

# Score : The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()

reel = ks.Series(y_test).to_frame().rename(columns = {0:'Reel'})
result = ks.concat([df,reel],axis=1)

result['square_diff_true_pred'] = (result['Reel'] - result['prediction']) ** 2
u = result['square_diff_true_pred'].sum()
v = ((result['Reel'] - result['Reel'].mean()) ** 2).sum()

score = (1 - u/v)
print(f"score: {score}")

# %% [markdown]
# ## Entrainement et inférence avec Pipeline
# %% [markdown]
# Seuls les modèles entrainés et les prédictions peuvent être utilisés avec koalas

# %%



