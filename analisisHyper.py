import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
data = pd.read_csv('Customer-Churn-Records.csv')

# Ver los primeros registros
#print('dataset head')
#print(data.head())

# Ver información general sobre el dataset
#print('data info')
#print(data.info())

# Ver estadísticas descriptivas
#print('descripcion de los datos')
#print(data.describe())

# Eliminar columnas innecesarias
data = data.drop(columns=['RowNumber', 'Surname', 'CustomerId'])

# Verificar valores nulos
#print('valores nulos?')
#print(data.isnull().sum())

# Convertir 'Exited' a categoría (si no lo está)
data['Exited'] = data['Exited'].astype('category')

# Codificación de variables categóricas
data = pd.get_dummies(data, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)


X = data.drop('Exited', axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)


grid_search.fit(X_train, y_train)

print('Mejores hiperparámetros:', grid_search.best_params_)

# Usar el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Realizar predicciones
y_pred = best_model.predict(X_test)

# Evaluar el modelo
print('Matriz de Confusión:')
print(confusion_matrix(y_test, y_pred))

print('Informe de Clasificación:')
print(classification_report(y_test, y_pred))

# Visualizar la importancia de las características
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]