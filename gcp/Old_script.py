# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from datetime import datetime

# %% [markdown]
# # Chargement des dataset
# %% [markdown]
# ## YouGov - Wearing Mask in public
# 

# %%
start = datetime.now()


##Chargement dataset
df = pd.read_csv(
    "gs://dask-vs-koalas/wearing_face_mask_public.csv",
    sep=";"
)



## Transformation du dataset = 1 ligne par date/pays
# On ne garde qu'une seule valeur par jour : le max pour chaque pays
format = '%Y-%m-%d %H:%M:%S'
df['DateTime'] = pd.to_datetime(df['DateTime'], format=format)
df['DateTime'] = df['DateTime'].dt.normalize()

df = df.sort_values('DateTime').groupby(df['DateTime']).max()
df = df.set_index(pd.DatetimeIndex(df['DateTime'])).drop(['DateTime'], axis=1)
wearing_mask_in_public_data = df.resample('1D').pad()
wearing_mask_in_public_data = wearing_mask_in_public_data.fillna(0)
wearing_mask_in_public_data = wearing_mask_in_public_data.reset_index().melt(
                                id_vars=['DateTime'], 
                                var_name='country', 
                                value_name='percent_wearing_mask')


print(f"Le dataset contient {len(df)} enregistrements")

print("Sample dataset final:")
print(wearing_mask_in_public_data.sample(5))

stop = datetime.now()

print("Temps de chargement et tranformation petit dataset : ", (stop-start).microseconds/1000, "ms")

# %% [markdown]
# ## Google - Covid 19 Open Data

# %%
start = datetime.now()

#Chargement dataset
covid19_opendata = pd.read_csv(
    "gs://dask-vs-koalas/main.csv",
    keep_default_na=False,
    na_values=[""])



# Jointure entre open data covid 19 et yougo
covid19_opendata['date'] = pd.to_datetime(covid19_opendata['date'], format=format)
covid19_merge1 = covid19_opendata.merge(wearing_mask_in_public_data, 
                                      left_on = ['country_name','date'],
                                      right_on = ['country','DateTime'], how = 'left')


remove_cols = ['key', 'country','aggregation_level','locality_code', 'wikidata', 'datacommons', 'country_code', 'subregion1_code', 'subregion1_name', 'subregion2_code', 'subregion2_name', 'locality_name', '3166-1-alpha-2', '3166-1-alpha-3', 'DateTime']

covid19_merge1 = covid19_merge1.drop(remove_cols, axis=1)
covid19_merge1 = covid19_merge1.fillna(0)

prepared_data =  covid19_merge1.copy()

## Encode Pays
from sklearn.preprocessing import LabelEncoder

encoded_countries = LabelEncoder().fit_transform(prepared_data.country_name)
prepared_data['country_name'] = encoded_countries

## Encode Date
dates = prepared_data.date.apply(lambda x: x.strftime('%Y%m%d'))
encoded_dates = LabelEncoder().fit_transform(dates)
prepared_data['date'] = encoded_dates

print(f"Le dataset contient {len(prepared_data)} enregistrements")

print("Sample dataset final:")
print(prepared_data.sample(5))

stop = datetime.now()


print("Temps de chargement et tranformation grand dataset : ", (stop-start).seconds, "s")

# %% [markdown]
# ## Entrainement et inférence

# %%
start = datetime.now()

# Split Train/Test
from sklearn.model_selection import train_test_split
X = prepared_data.loc[:, prepared_data.columns != 'new_confirmed']
y = prepared_data['new_confirmed'].ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Scale des valeurs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Entraintement MLP
from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(max_iter=10, hidden_layer_sizes=(100, 50, 25, 10, 5), verbose=True).fit(X_train, y_train)

# Prédiction et Score
score = regr.score(X_test, y_test)

stop = datetime.now()

print("Temps préparation et inférence (ML) : ", (stop-start).seconds, "s")
print(f"model score: {score}")

# %% [markdown]
# ## Entrainement et inférence avec Pipeline

# %%
start = datetime.now()

# Split Train/Test
from sklearn.model_selection import train_test_split
X = prepared_data.loc[:, prepared_data.columns != 'new_confirmed']
y = prepared_data['new_confirmed'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.pipeline import Pipeline


# Scale des valeurs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Entraintement MLP
from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(max_iter=10, hidden_layer_sizes=(100, 50, 25, 10, 5), verbose=True)

pipeline = Pipeline([('scaler', scaler), ('regressor', regr)])

# Exécution du pipeline
pipeline.fit(X_train, y_train)

# Prédiction et Score
score = pipeline.score(X_test, y_test)

stop = datetime.now()

print("Temps préparation et inférence (ML) : ", (stop-start).seconds, "s")
print(f"model score: {score}")


# %%



