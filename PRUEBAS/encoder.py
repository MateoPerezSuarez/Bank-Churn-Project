import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
'''We have'''
df = pd.read_csv('Bank-Churn-Project/src/rawDataset.csv')
df.drop(columns=['RowNumber','Surname','CustomerId'], inplace= True)

'''
columnas = df.columns
x = df.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns = columnas)

print(df.head())
df.to_csv('src/normalDataset.csv', index = False)
'''

'''OPCION 1'''
categoricalCols = df.select_dtypes(include=['object']).columns
categoricalCols.drop('Gender')
encoder = OneHotEncoder()

for col in categoricalCols:
    df[col]  = encoder.fit_transform(df[col])
print(df.head())


'''OPCION 2
encoder = LabelEncoder()
categoricalCols = df.select_dtypes(include=['object']).columns

for col in categoricalCols:
    df[col] = encoder.fit_transform(df[col])
    print(f"Column {col} changed")
print(df.head())
'''