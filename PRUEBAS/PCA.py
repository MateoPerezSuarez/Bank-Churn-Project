import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


df = pd.read_csv('src/rawDataset.csv')
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
encoder = OneHotEncoder(sparse_output=False, drop='first')

catcols = ['Gender','Geography', 'Card Type']
encData = encoder.fit_transform(df[catcols])
encDF = pd.DataFrame(encData, columns=encoder.get_feature_names_out(catcols))
df = pd.concat([df.drop(columns= catcols),encDF],axis=1)
'''
X, Y = df.drop('Exited',axis =1), df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
ridge_reg = Ridge(alpha=5)
ridge_reg.fit(X_train, y_train)
y_pred = ridge_reg.predict(X_test)



y_pred_class = (y_pred >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"ROC-AUC: {roc_auc}")
'''
df.to_csv('encodedData.csv',index=False)
nums = ['CreditScore', 'Age', 'EstimatedSalary', 'Point Earned']
scaler = StandardScaler()
standData = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

'''
X = df[nums]
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


pca = PCA(n_components= 0.95)
pcaData = pca.fit_transform(standData)
print(pcaData)

variance = pca.explained_variance_ratio_
nComponents = pca.n_components_

print(variance)
print(nComponents)
'''

'''TO EXPLORE WHICH ARE THE BEST PCA VALUES'''
'''PRUEBA REGULARIZATION'''

'''PCA NUMBER OF COMPONENTS CHANGER Selecter'''



scaler = StandardScaler()
standData = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
pca = PCA(n_components= 0.95)
pcaData = pca.fit_transform(standData)
