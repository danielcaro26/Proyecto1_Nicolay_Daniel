import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy import stats

df = pd.read_csv("SeoulBikeData_utf8.csv")
print(df.head())
print("\n", df.shape)

print("\n", df["Brand"].unique())

print("\n", df.groupby("Brand").count())

print("\n", df["Brand"].value_counts())

print("\n", df.describe())


# Número de duplicados
duplicates = len(df[df.duplicated()])
print("\n", f'Number of Duplicate Entries: {duplicates}')

# Número de valores perdidos
missing_values = df.isnull().sum().sum()
print("\n", f'Number of Missing Values: {missing_values}')