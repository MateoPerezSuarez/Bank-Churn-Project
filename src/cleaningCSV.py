import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('src/rawDataset.csv')
print(df.head())

'''Columns rowNumber, Surname and CustomerId are not useful to train our models in next stages of the projects so we have decided to delete them.
Surname is not necessary as we have a column ClientID which is the unique ID from each of the customers, so another identification column is not necessary for us'''

df.drop(columns=['RowNumber','Surname','CustomerId'], inplace= True)


dummy_columns = ['Gender','Geography','Card Type']
for col in dummy_columns:
        df_dummies = pd.factorize(df[col])[0]
        df[col] = df_dummies


print(df.head())
df.to_csv('src/cleanDataset.csv', index = False)










