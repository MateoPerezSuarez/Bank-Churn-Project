import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
'''We have'''
df = pd.read_csv('src/rawDataset.csv')
df.drop(columns=['RowNumber','Surname','CustomerId'], inplace= True)


columnas = df.columns
x = df.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns = columnas)



print(df.head())
df.to_csv('src/normalDataset.csv', index = False)


'''OPCION 1'''
columns_to_change = df.select_dtypes(include = "object").columns
print(columns_to_change)
'''
encoder = OneHotEncoder(drop= "first")
x = encoder.fit_transform(df[columns_to_change]).toarray()
x = pd.DataFrame(x, columns = encoder.get_feature_names_out())
x = x.astype(int)
temp = df.drop(columns_to_change, axis=1)
df = pd.concat([temp,x], axis=1)
'''

'''OPCION 2

encoder = LabelEncoder()

for col in columns_to_change:
    df[col] = encoder.fit_transform(df[col])
    print(f"Column {col} changed successfully")

print(df.head())
'''

