# 🏦 Bank Customer Churn Prediction

A *Machine Learning* project focused on **predicting customer churn** in a banking institution, using exploratory data analysis, feature engineering, classification models, regression, clustering, and model interpretability techniques.

---

## 👥 Authors
- Mateo Pérez  
- Iñigo Peña  
- Gotzon Viteri  
- Josu Viteri  

---

## 🎯 Project Objective

The main goal of this project is to **predict which customers are most likely to leave the bank (churn)** by analyzing their behavior and financial characteristics.  
This enables banks to anticipate customer attrition and design more effective retention strategies.

Additionally, the project places strong emphasis on:
- Model interpretability  
- Critical data analysis  
- Comparison of different *Machine Learning* approaches  

---

## 📊 Dataset

- **Source**: Kaggle – Bank Customer Churn Dataset  
- **Records**: 10,000 customers  
- **Features**: 18 (numerical, categorical, and boolean)

### Key variables
- **Numerical**: Age, CreditScore, Balance, Tenure, EstimatedSalary  
- **Categorical**: Geography, Gender, Card Type  
- **Boolean**: HasCrCard, IsActiveMember, Exited (target variable)

---

## 🧪 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Distribution analysis  
- Class imbalance detection  
- Correlation analysis  
- Advanced visualizations (heatmaps, KDEs, pairplots)

**Key findings**:
- Strong class imbalance in the `Exited` variable  
- Very high correlation between `Complain` and `Exited`  
- Non-active customers and customers with multiple products show higher churn rates  

---

### 2️⃣ Feature Engineering
- Removal of irrelevant identifiers  
- Age discretization  
- Binarization of the number of products  
- One-Hot Encoding  
- Feature scaling  
- Dimensionality reduction using PCA  

---

### 3️⃣ Classification Models
The following models were trained and compared:
- Logistic Regression  
- Decision Tree (CART)  
- Random Forest  
- Support Vector Machines (SVC)  
- Naive Bayes  

**Best-performing model**: 🌲 **Random Forest Classifier**

**Key metrics**:
- Accuracy: ~76%  
- Recall: ~70%  
- ROC-AUC: ~82%  

Recall was prioritized due to its importance in identifying customers at risk of churn.

---

### 4️⃣ Regression
- Target variable: `Balance`  
- Models tested: Linear Regression, GLM, SVR, Random Forest Regressor, among others  

⚠️ Results were limited due to:
- Extreme outliers  
- Dataset not being well-suited for regression tasks  

---

### 5️⃣ Clustering
- Hyperparameter tuning  
- Model comparison  
- Cluster visualization and interpretation  
- Identification of customer subgroups  

---

### 6️⃣ Interpretability
- Feature importance analysis  
- Partial Dependence Plots (PDP)  
- ALE plots  
- SHAP values (executed in Google Colab due to compatibility issues)  

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Libraries**:
  - pandas, numpy  
  - matplotlib, seaborn, plotly  
  - scikit-learn  
  - optuna  
  - shap, alibi (Google Colab)  


---

## 📌 Conclusions

- **Class imbalance** and **dataset limitations** negatively affect model performance.  
- Random Forest provides the best trade-off between evaluation metrics.  
- Model interpretability is essential to deliver real business value.  
- A richer and more balanced dataset could significantly improve results.  

---

## 🚀 Future Improvements
- Oversampling techniques (e.g., SMOTE)  
- Deep Learning approaches  
- Additional behavioral features  
- More extensive hyperparameter tuning  

---

## 📄 License
Academic project developed for educational purposes.



