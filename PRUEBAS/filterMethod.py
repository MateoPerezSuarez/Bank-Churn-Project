import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import seaborn as sns
from sklearn.pipeline import Pipeline



df = pd.read_csv('src/encodedData.csv').iloc[:,1:-1]
targetCorr = df.corr()[['Exited']]
'''Correlation Plot
plt.figure(figsize=(7,5))
sns.heatmap(targetCorr, annot=True, cmap = plt.cm.Blues)
plt.show()

mostCorr = targetCorr[abs(targetCorr)>0.5].dropna()
print(mostCorr)
'''


'''LINEAR ANALYSIS'''
'''model that finds linear combinations of the features that achieve:
-max separability between classes
-min variance within each class.
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


'''Correlation Coefficient'''
corr_matrix = df.corr()
bestFeatures = corr_matrix.index[abs(corr_matrix['Exited']) >0.1]
dfFiltered = df[bestFeatures]
print(dfFiltered)


'''Chi-Square'''
X = df.drop('Exited', axis=1)
y = df['Exited']

sets = SelectKBest(chi2, k=8)
X_new = sets.fit_transform(X,y)

selectedCols = X.columns[sets.get_support()]
print(selectedCols)

'''ANNOVA TEST'''
fval, pval = f_classif(X,y)
sets = SelectKBest(score_func=f_classif, k=8)
ressAnova = pd.DataFrame({'Feature': X.columns, 'F-Value': fval, 'P-Value': pval})
print(ressAnova.sort_values(by='P-Value'))
importantFeatures = ressAnova[ressAnova['P-Value'] <0.05]['Feature']
df_filter = df[importantFeatures]

'''EXPLANATION: 
        -F-Value is an statistical measure  to compare the variability between groups
        with the variability within groups. So F-Value = Variation between groups / Variation within groups.


        -The P-value explains the probability to obtain an F-value as the one observed if the null hypothesis is true.
        So the null hypothesis for Annova stablishes that there are no significative differences between groups means.


        In our case we obtained that there are 
'''
fs = SelectKBest(score_func=f_classif, k=8)
X_new = fs.fit(X,y)









'''
print(y.value_counts())

steps = [('lda', LinearDiscriminantAnalysis()),
         ('scaler', StandardScaler()),
         ('logreg', LogisticRegression(C=10, max_iter=200,solver='lbfgs', class_weight='balanced'))
        ]


modelLDA = Pipeline(steps=steps)

modelNoLda = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(C=10, max_iter=200, solver= 'lbfgs', class_weight='balanced'))
    ])

cv = StratifiedKFold(n_splits = 5)

With LDA
n_scores_lda = cross_val_score(modelLDA, X, y ,scoring= 'f1_macro', cv=cv, n_jobs=-1)
lregr = LogisticRegression(C=10)


n_scores_NoLDA = cross_val_score(modelNoLda, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

Intentar F1 score para chequear la diferencia entre usar o no el LDA
accuracy = cross_val_score(modelLDA, X, y, scoring='accuracy', cv = cv, n_jobs = -1)
precision = cross_val_score(modelLDA, X,y,scoring='precision', cv=cv, n_jobs =-1)
recall = cross_val_score(modelLDA, X,y, scoring='recall', cv=cv, n_jobs=-1)
print('Without LDA: %.2f' % np.mean(n_scores_lda))
print('With LDA: %.2f' % np.mean(n_scores_NoLDA))

print('Accuracy: %.2f' % np.mean(accuracy))
print('Precision: %.2f' % np.mean(precision))
print('Recall: %.2f' % np.mean(recall))



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelLDA.fit(X_train, y_train)
y_pred = modelLDA.predict(X_test)
print(classification_report(y_test, y_pred))
'''
''''''










