import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt
from sklearn import decomposition as PCA
from sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

bank_Data = pd.read_csv('Churn_Modelling.csv')
bank_Data.fillna(value=0)
#print(bank_Data)

#Valores nulos 

dataLimpio = bank_Data.dropna()
dataLimpio = dataLimpio.drop_duplicates()

#Contar Repetidos
print(sum(dataLimpio.duplicated()))
print(dataLimpio.isnull().sum())


columnasRecom = ['Age','CreditScore', 'EstimatedSalary','Balance']
x = dataLimpio.loc[:,columnasRecom]
x = StandardScaler().fit_transform(x)
standarization = pd.DataFrame(data = x, columns = columnasRecom)
print(standarization)

#PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pcDataframe = pd.DataFrame(data = principalComponents, columns=['principal comp 1', 'principal comp 2'] )
print(pcDataframe)