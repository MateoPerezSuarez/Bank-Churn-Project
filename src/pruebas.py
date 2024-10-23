import pandas as pd
from sklearn import preprocessing


'''We have'''
df = pd.read_csv('src\cleanDataset.csv')
print(df.head())

columnas = df.columns
x = df.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns = columnas)

print(df.head())
df.to_csv('src/normalDataset.csv', index = False)


