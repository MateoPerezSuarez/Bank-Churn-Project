import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a1bfc70 (LA LLORONA 1)
=======
>>>>>>> FSbranch
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> FSbranch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import GaussianNB
<<<<<<< HEAD
=======
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
=======
>>>>>>> a1bfc70 (LA LLORONA 1)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, f_classif
<<<<<<< HEAD
import seaborn as sns
from sklearn.pipeline import Pipeline
>>>>>>> f8e4345 (PRUEBAS DE FILTER METHODS)
=======
from sklearn.naive_bayes import GaussianNB
>>>>>>> a1bfc70 (LA LLORONA 1)
=======
>>>>>>> FSbranch



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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
=======
>>>>>>> FSbranch
'''Correlation Coefficient'''
corr_matrix = df.corr()
bestFeatures = corr_matrix.index[abs(corr_matrix['Exited']) >0.1]
dfFiltered = df[bestFeatures]
print(dfFiltered)


'''Chi-Square'''
X = df.drop(['Exited','Complain'], axis=1)
y = df['Exited']

chiSqSet = SelectKBest(chi2, k=14)
X_new = chiSqSet.fit_transform(X,y)

selectedCols = X.columns[chiSqSet.get_support()]
print(selectedCols)
'''TRAINING WITH CHI-SQUARE'''
#CON RANDOMFOREST
'''
testParam = {
    'n_estimators': [100,200,300,500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'class_weight':['balanced',None]

}
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42,stratify=y)



gridSrc = GridSearchCV(estimator = RandomForestClassifier(random_state=42),param_grid=testParam,cv=5, n_jobs = -1, verbose=1, scoring= 'accuracy')

bestEst = gridSrc.best_estimator_
bestPar = gridSrc.best_params_

yPred = bestEst.predict(X_test)
yProba = bestEst.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, yPred)
roc_auc = roc_auc_score(y_test,yProba)
report = classification_report(y_test,yPred)
confMatrix = confusion_matrix(y_test,yPred)

print(f"THE BEST PARAMETERS FOR RANDOM FOREST ARE: {bestPar}")
'''


#CON RANDOMFOREST

selectedFeatures = selectedCols.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size= 0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

yPred = model.predict(X_test)
accuracyScore = accuracy_score(y_test, yPred)
report = classification_report(y_test, yPred)

print("\nRandom Forest RESULTS:")
print(report)
print(f'Accuracy_ {accuracyScore}')
print("-"*60)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(X_train)
xTestScaled = scaler.fit_transform(X_test)

#LOGISTIC REGRESSION AND NAIVE BAYES

models = {
    "Naive Bayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "SVM": svm.SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
}

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(xTrainScaled,y_train)
    yPred =model.predict(xTestScaled)
    yProba =model.predict_proba(xTestScaled)[:,1]
    report  =classification_report(y_test, yPred)
    roc_auc = roc_auc_score(y_test, yProba)
    print(f"\nResults of {name}:")
    print(report)
    print(f"ROC-AUC: {roc_auc:.2f}")
    print("-"*60)

'''ANNOVA TEST'''
fval, pval = f_classif(X,y)
sets = SelectKBest(score_func=f_classif, k=8)
ressAnova = pd.DataFrame({'Feature': X.columns, 'F-Value': fval, 'P-Value': pval})
print(ressAnova.sort_values(by='P-Value'))
importantFeatures = ressAnova[ressAnova['P-Value'] <0.05]['Feature']
df_filter = df[importantFeatures]

print(df_filter)


'''EXPLANATION: 
        -F-Value is an statistical measure  to compare the variability between groups
        with the variability within groups. So F-Value = Variation between groups / Variation within groups.


        -The P-value explains the probability to obtain an F-value as the one observed if the null hypothesis is true.
        So the null hypothesis for Annova stablishes that there are no significative differences between groups means.


        In our case we obtained that there are 
'''
fs = SelectKBest(score_func=f_classif, k=8)
X_new = fs.fit(X,y)









<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 42401c4 (...d.ad.)
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



<<<<<<< HEAD
>>>>>>> f8e4345 (PRUEBAS DE FILTER METHODS)
'''Correlation Coefficient'''
corr_matrix = df.corr()
bestFeatures = corr_matrix.index[abs(corr_matrix['Exited']) >0.1]
dfFiltered = df[bestFeatures]
print(dfFiltered)


'''Chi-Square'''
<<<<<<< HEAD
X = df.drop(['Exited','Complain'], axis=1)
y = df['Exited']

chiSqSet = SelectKBest(chi2, k=14)
X_new = chiSqSet.fit_transform(X,y)

selectedCols = X.columns[chiSqSet.get_support()]
print(selectedCols)
'''TRAINING WITH CHI-SQUARE'''
#CON RANDOMFOREST
'''
testParam = {
    'n_estimators': [100,200,300,500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'class_weight':['balanced',None]

}
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42,stratify=y)



gridSrc = GridSearchCV(estimator = RandomForestClassifier(random_state=42),param_grid=testParam,cv=5, n_jobs = -1, verbose=1, scoring= 'accuracy')

bestEst = gridSrc.best_estimator_
bestPar = gridSrc.best_params_

yPred = bestEst.predict(X_test)
yProba = bestEst.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, yPred)
roc_auc = roc_auc_score(y_test,yProba)
report = classification_report(y_test,yPred)
confMatrix = confusion_matrix(y_test,yPred)

print(f"THE BEST PARAMETERS FOR RANDOM FOREST ARE: {bestPar}")
'''


#CON RANDOMFOREST

selectedFeatures = selectedCols.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size= 0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

yPred = model.predict(X_test)
accuracyScore = accuracy_score(y_test, yPred)
report = classification_report(y_test, yPred)

print("\nRandom Forest RESULTS:")
print(report)
print(f'Accuracy_ {accuracyScore}')
print("-"*60)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(X_train)
xTestScaled = scaler.fit_transform(X_test)

#LOGISTIC REGRESSION AND NAIVE BAYES

models = {
    "Naive Bayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "SVM": svm.SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
}

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(xTrainScaled,y_train)
    yPred =model.predict(xTestScaled)
    yProba =model.predict_proba(xTestScaled)[:,1]
    report  =classification_report(y_test, yPred)
    roc_auc = roc_auc_score(y_test, yProba)
    print(f"\nResults of {name}:")
    print(report)
    print(f"ROC-AUC: {roc_auc:.2f}")
    print("-"*60)
=======
X = df.drop('Exited', axis=1)
y = df['Exited']

sets = SelectKBest(chi2, k=8)
X_new = sets.fit_transform(X,y)

selectedCols = X.columns[sets.get_support()]
print(selectedCols)
=======


>>>>>>> 42401c4 (...d.ad.)

>>>>>>> f8e4345 (PRUEBAS DE FILTER METHODS)

<<<<<<< HEAD
'''ANNOVA TEST'''
fval, pval = f_classif(X,y)
sets = SelectKBest(score_func=f_classif, k=8)
ressAnova = pd.DataFrame({'Feature': X.columns, 'F-Value': fval, 'P-Value': pval})
print(ressAnova.sort_values(by='P-Value'))
importantFeatures = ressAnova[ressAnova['P-Value'] <0.05]['Feature']
df_filter = df[importantFeatures]

<<<<<<< HEAD
print(df_filter)


=======
>>>>>>> f8e4345 (PRUEBAS DE FILTER METHODS)
'''EXPLANATION: 
        -F-Value is an statistical measure  to compare the variability between groups
        with the variability within groups. So F-Value = Variation between groups / Variation within groups.
=======
>>>>>>> 42401c4 (...d.ad.)



<<<<<<< HEAD






=======
>>>>>>> a1bfc70 (LA LLORONA 1)


=======
>>>>>>> f8e4345 (PRUEBAS DE FILTER METHODS)
=======

>>>>>>> FSbranch
