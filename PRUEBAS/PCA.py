import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Bank-Churn-Project/src/rawDataset.csv')
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)

encoder = OneHotEncoder(sparse_output=False, drop='first')

catcols = ['Gender','Geography', 'Card Type']
encData = encoder.fit_transform(df[catcols])
encDF = pd.DataFrame(encData, columns=encoder.get_feature_names_out(catcols))
df = pd.concat([df.drop(columns= catcols),encDF],axis=1)



scaler = StandardScaler()
standData = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
pca = PCA(n_components= 6)
pcaData = pca.fit_transform(standData)

variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada vs Número de Componentes')
plt.grid()
plt.show()

print(pcaData)