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

# Número de duplicados
duplicates = len(df[df.duplicated()])
print("\n", f'Number of Duplicate Entries: {duplicates}')

# Número de valores perdidos
missing_values = df.isnull().sum().sum()
print("\n", f'Number of Missing Values: {missing_values}')

