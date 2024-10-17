import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
data = pd.read_csv('Customer-Churn-Records.csv')

# Eliminar columnas innecesarias
data = data.drop(columns=['RowNumber', 'Surname', 'CustomerId'])

# Convertir 'Exited' a categoría (si no lo está)
data['Exited'] = data['Exited'].astype('category')

# Codificación de variables categóricas
data = pd.get_dummies(data, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)

X = data.drop('Exited', axis=1)
y = data['Exited']

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest con los mejores hiperparámetros
model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42)

# Ajustar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print('Matriz de Confusión:')
print(confusion_matrix(y_test, y_pred))

print('Informe de Clasificación:')
print(classification_report(y_test, y_pred))