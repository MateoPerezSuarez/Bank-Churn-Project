
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv('Churn_Modelling.csv')

# Define numeric and categorical columns
numeric_cols = ['CreditScore', 'Balance', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categoric_cols = ['Geography', 'Gender']

# Remove rows with missing values in key columns
data_clean = data.dropna(subset=numeric_cols + categoric_cols + ['Exited'])


X = data_clean[numeric_cols + categoric_cols]
y = data_clean['Exited']

#dummies
X = pd.get_dummies(X, columns=categoric_cols, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest training

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

#results
print('Accuracy: ' + str(accuracy))
print('Confusion Matrix:\n' + str(conf_matrix))
print('Classification Report:\n' + str(class_report))

