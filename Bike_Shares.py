import numpy as np
import pandas as pd
import seaborn as sns
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

#Análisis descriptivo de los Datos

# Gráfico de líneas
sns.lineplot(x='Date', y='Rented Bike Count', data=df)
plt.title('Evolución del número de bicicletas rentadas')
plt.xlabel('Fecha')
plt.ylabel('Número de bicicletas rentadas')
plt.show()

#Histograma de Frecuencia
sns.histplot(data=df, x='Rented Bike Count')
plt.title("Distribución del número de bicicletas rentadas")
plt.xlabel("Número de bicicletas rentadas")
plt.ylabel("Frecuencia")
plt.show()

#Diagramas de Caja 
sns.boxplot(x='Functioning Day',y='Temperature(C)', data=df)
plt.title("Relación entre los días laborales y la temperatura")
plt.xlabel("día Laboral")
plt.ylabel("Temperatura")
plt.show()

#Diagramas de dispersión
sns.scatterplot(x='Temperature(C)', y='Solar Radiation (MJ/m2)', data=df)
plt.title("Relación entre Temperatura y Radiación solar")
plt.xlabel("Temperatura (C)")
plt.ylabel("Radiación Solar (MJ/m2)")
plt.show()

#Diagramas de violin
sns.violinplot(x='Holiday', y='Wind speed (m/s)', data=df)
plt.title("Relación entre los días festivos y la velocidad del viento")
plt.xlabel("Holiday")
plt.ylabel("Velocidad del viento")
plt.show()

#Histogramas por cada variable y dispersión entre variables
sns.pairplot(df,)

#Diagramas de dispersión y tendencia entre variables de entrada y variable de respuesta
sns.pairplot(df, x_vars=['Temperature(C)','Wind speed (m/s)','Visibility (10m)','Humidity(%)'], y_vars='Rented Bike Count', height=7, aspect=0.7, kind='reg')

#Matriz de Correlación 
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()



