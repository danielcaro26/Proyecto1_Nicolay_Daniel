import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy import stats

df = pd.read_csv("C:/Users/danis/Downloads/SeoulBikeData_utf8.csv")


# Información general del DataFrame
print(df.head())  # Primeras filas
print(df.tail())  # Últimas filas
print(df.shape)   # Número de filas y columnas
print(df.info())  # Resumen de tipos de datos y valores nulos
print(df.describe())  # Estadísticas descriptivas de las columnas numéricas

# Información sobre los tipos de datos
print(df.dtypes)
print(df.dtypes.value_counts())
print('Number of Features: %d'%(df.shape[1]))

# Número de duplicados
duplicates = len(df[df.duplicated()])
print("\n", f'Number of Duplicate Entries: {duplicates}')

# Número de valores perdidos
missing_values = df.isnull().sum().sum()
print("\n", f'Number of Missing Values: {missing_values}')

#Codificar variables categoricas

#Encontrar cuales columnas pueden ser categoricas 
catcols = df.select_dtypes(exclude = ['int64','float64']).columns
print(catcols)

#Convertir fecha en el formato adecuado
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

#Encontrar valores unicos de cada columna categorica
print(df["Seasons"].unique())
print(df["Holiday"].unique())
print(df["Functioning Day"].unique())

#Información de las columnas categoricas 
print(df.groupby(["Seasons"]).count())
print(df.groupby(["Holiday"]).count())
print(df.groupby(["Functioning Day"]).count())

# Conversión de características en las columnas "Holiday" y "Functioning Day"
df['Holiday'] = df['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
df['Functioning Day'] = df['Functioning Day'].map({'Yes': 1, 'No': 0})

#Codificar valores de la columna "Seasons"
df = pd.get_dummies(df, columns = ["Seasons"])

#Verificar el cambio de las columnas categoricas
# Información sobre los tipos de datos
print(df.dtypes)
print(df.dtypes.value_counts())
print('New Number of Features: %d'%(df.shape[1])) 


