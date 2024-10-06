import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#Check if dataset has NaN values or null where there might not be
df = pd.read_csv('Churn_Modelling.csv')
print(df.isnull().sum())
print((df ==0).sum())

colNames = ['Geography', 'Age', 'HasCrCard', 'IsActiveMember']
for col in colNames:
    print(df[df[col].isnull()])

