import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Churn_Modelling.csv')


numeric_cols = ['CreditScore', 'Balance', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_cols = ['Geography', 'Gender']

#As we have some rows with NaN values we should not take them into account for a proper working of the function
data_train = data.dropna(subset=numeric_cols + categorical_cols + ['Age'])

X = data_train[numeric_cols + categorical_cols]
y = data_train['Age']

#Need to obtain the dummy variables for both of the categorical columns that we are using
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


model = LinearRegression()
model.fit(X, y)

missing_Value = data[data['Age'].isna()][numeric_cols + categorical_cols]
missing_Value = pd.get_dummies(missing_Value, columns=categorical_cols, drop_first=True)
missing_Value = missing_Value.reindex(columns=X.columns, fill_value=0)

age_predicted = model.predict(missing_Value)
print(age_predicted[0])
data.loc[data['Age'].isna(), 'Age'] = age_predicted[0]

#print(data.describe())

