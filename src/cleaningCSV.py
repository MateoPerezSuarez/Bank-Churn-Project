import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('src/rawDataset.csv')
print(df.head())

'''Columns rowNumber, Surname and CustomerId are not useful to train our models in next stages of the projects so we have decided to delete them.
Surname is not necessary as we have a column ClientID which is the unique ID from each of the customers, so another identification column is not necessary for us'''

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
# import category_encoders as ce
# encoder = ce.TargetEncoder(cols=['nombre_de_la_variable'])
# df['nombre_de_la_variable_encoded'] = encoder.fit_transform(df['nombre_de_la_variable'], df['target'])




print(df.head())
df.to_csv('src/cleanDataset.csv', index = False)




