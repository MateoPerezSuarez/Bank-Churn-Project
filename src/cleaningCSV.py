import pandas as pd

df = pd.read_csv('src/rawDataset.csv')
print(df.head())

dummy_columns = ['Gender','Geography','Card Type']

for col in dummy_columns:
    if df[col].nunique()  > 2:
        df_dummies = pd.factorize(df[col])[0]
        df[col] = df_dummies
        print("Column Changed successfully")
    else:
        df_dummies = pd.get_dummies(df[col], drop_first= True, dtype = int)
        df[col] = df_dummies
        print("Column Changed successfully")

print(df.head())
df.to_csv('src/CleanDataset.csv', index = False)



