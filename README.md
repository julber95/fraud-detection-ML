# Fraud Detection using Machine Learning

**A machine learning project designed to detect fraudulent transactions in customer purchases.**  
The goal is to optimize the **Precision-Recall AUC (PR-AUC)** to accurately identify fraudulent transactions while minimizing false positives.

---

## 📌 Problem Statement
Fraud detection in banking transactions is a major challenge. This project focuses on analyzing transaction data and training classification models to detect fraudulent activities.

---

## 📊 Dataset Overview
- **Number of samples**: 92,790
- **Number of features**: 146
- **Target variable**: `fraud_flag` (1 = Fraudulent transaction, 0 = Non-fraudulent)
- **Feature types**: 94 categorical, 52 numerical

---

## 🔧 Data Preprocessing
1. **Data Cleaning & Exploration**
   - Identified missing values and visualized them using heatmaps
   - Removed irrelevant columns such as `goods_code`
   - Merged duplicate categorical columns for better feature representation

2. **Categorical Variable Encoding**
   - 🌟 **One-Hot Encoding** instead of Label Encoding to prevent bias

3. **Handling Missing Values**
   - Filled missing values in `cash_price` and `Nbr_of_prod_purchas` using `.fillna(False)`

---

## 🚀 Machine Learning Models
Multiple models were tested using **Precision-Recall AUC (PR-AUC)** as the evaluation metric.

| Model | PR-AUC Score |
|--------|------------|
| Decision Tree | 0.07 |
| Logistic Regression | 0.01 |
| Random Forest | 0.22 |
| **Gradient Boosting (Best Model)** | **0.2813** |

- 🌟 **Hyperparameter optimization using RandomSearchCV**
- 📌 **Cross-validation** for better model robustness
