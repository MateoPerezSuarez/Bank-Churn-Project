import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib import pyplot as plt

df = pd.read_csv("src/cleanDataset.csv")

'''Comprobations to check if there are any missing values or repeated columns'''
print(df.isna().sum())
print(f"There are: {df.duplicated().sum()} duplicated rows")


'''Explanation:
    We are going to make some plotting tasks to analyze our dataset,
    to have a clear overview of our dataset, such as: Distribution of data, if it is balanced or not,...'''
    
plt.figure(figsize= (5,5))
plot_data = (df[['Exited']].value_counts())
plt.pie(plot_data, labels=['Not exited', 'Exited'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Exited percentages')
plt.show()



'''Corre'''
corr_matrix = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']].corr()
sns.heatmap(corr_matrix, annot=True, cmap= "Spectral")
plt.show()


