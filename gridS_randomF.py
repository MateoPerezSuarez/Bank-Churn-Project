import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('Churn_Modelling.csv')

# Define numeric and categorical columns
numeric_cols = ['CreditScore', 'Balance', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categoric_cols = ['Geography', 'Gender']

# Remove rows with missing values in key columns
data_clean = data.dropna(subset=numeric_cols + categoric_cols + ['Exited'])

# Feature selection
X = data_clean[numeric_cols + categoric_cols]
y = data_clean['Exited']

# Convert categorical variables to dummy/one-hot encoding
X = pd.get_dummies(X, columns=categoric_cols, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)

# Set the hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Implement GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters found by GridSearchCV
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Best score found by GridSearchCV
best_cv_score = grid_search.best_score_
print(f'Best Cross-Validation Accuracy: {best_cv_score:.4f}')

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
