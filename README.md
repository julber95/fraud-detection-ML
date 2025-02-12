# Fraud Detection using Machine Learning

**A machine learning project designed to detect fraudulent transactions in customer purchases.**  
The goal is to optimize the **Precision-Recall AUC (PR-AUC)** to accurately identify fraudulent transactions while minimizing false positives.

---

## Context

BNP Paribas Personal Finance is the **leading consumer credit provider in France and Europe**, operating in over 30 countries with brands like **Cetelem, Cofinoga, and Findomestic**. The company offers a range of credit solutions through partnerships with **retailers, car manufacturers, financial institutions, and online merchants**.

As fraud becomes a growing concern for financial institutions, **fraud detection strategies** are a key component of BNP Paribas Personal Finance's risk management framework. The **Credit Process Optimization team** is responsible for improving credit decision-making, including fraud detection, by leveraging **advanced data analytics and machine learning models**.

---

## 👥 Team

This project was conducted as part of our ** Machine Learning project** in the second year of our **double bachelor's degree in Artificial Intelligence and Organizational Sciences** at **Paris Dauphine University**.

I had the opportunity to collaborate with the following teammates on this project:  

- **JIN Clémence**  
- **MOUANGUE Cameron**  
- **CHEN Franck**  

🏆 **Proud to be ranked 1st in both Public and Private Leaderboards!**

---

## Objective

The objective of this project is to develop a **fraud detection model** using machine learning techniques to analyze transaction basket data from one of our partners. The goal is to identify fraudulent transactions **before** they are approved, enabling proactive fraud prevention.

Given the **low fraud occurrence** (1.4% fraud cases in the dataset), the main challenge is to develop a model that **maximizes Precision-Recall AUC (PR-AUC)** while handling imbalanced data.

Our team achieved the best performance in the challenge!
- **🥇 1st place in the Public Leaderboard with a PR-AUC score of 0.2813**  
- **🥇 1st place in the Private Leaderboard with a PR-AUC score of 0.3507**  

🔗 **Challenge link:** [BNP Paribas Fraud Detection Challenge](https://challengedata.ens.fr/participants/challenges/104/)

---

## Dataset Overview

The dataset consists of **115,988 transactions**, where each observation represents a **basket of purchased items**. The dataset is divided into **training (80%) and test (20%) sets**.

- **Training set**: 92,790 transactions  
  - Fraudulent: **1,319** (1.42%)  
  - Non-fraudulent: **91,471**  

- **Test set**: 23,198 transactions  
  - Fraudulent: **362**  
  - Non-fraudulent: **22,836**  

**Target Variable (`fraud_flag`)**:  
- `1` → Fraudulent transaction  
- `0` → Legitimate transaction  

### 🔹 **Feature Description**
The dataset contains **147 columns**, which can be categorized as follows:

| Feature Category | Description |
|-----------------|-------------|
| `ID` | Unique transaction identifier |
| `item1` to `item24` | Product category (e.g., "Computer") |
| `cash_price1` to `cash_price24` | Price of each item |
| `make1` to `make24` | Manufacturer (e.g., "Apple") |
| `model1` to `model24` | Product model description |
| `goods_code1` to `goods_code24` | Product code |
| `Nbr_of_prod_purchas1` to `Nbr_of_prod_purchas24` | Number of units purchased |
| `Nb_of_items` | Total number of unique items in the basket |

Given the structure of the dataset, an important preprocessing step is to **flatten and aggregate** the item-level data to create meaningful features for the model.

---

## 🔧 Data Preprocessing

### **1️⃣ Data Cleaning & Feature Engineering**
- **Handled missing values** using heatmap analysis to detect patterns
- **Removed irrelevant columns** (`goods_code`) as they do not contribute to fraud detection
- **Merged duplicate categorical variables** to simplify basket representation
- **Aggregated basket-level data** to reduce dimensionality

### **2️⃣ Encoding Categorical Variables**
- Used **One-Hot Encoding** instead of Label Encoding to prevent introducing artificial ordinal relationships

### **3️⃣ Handling Missing Values**
- Imputed missing values in `cash_price` and `Nbr_of_prod_purchas` using `.fillna(False)`

### **4️⃣ Data Splitting**
- **80% training set**, **20% test set**, ensuring the same fraud distribution in both sets

---

## Machine Learning Models

We evaluated multiple classification models using **Precision-Recall AUC (PR-AUC)** as the primary metric due to the **class imbalance** in the dataset.

| Model | PR-AUC Score |
|--------|------------|
| Decision Tree | 0.07 |
| Logistic Regression | 0.01 |
| Random Forest | 0.22 |
| **Gradient Boosting (Best Model)** | **0.2813** |

### **Model Selection & Optimization**
- **Random Forest Classifier** was the most effective standard model.
- **Gradient Boosting Classifier** outperformed all models after hyperparameter tuning.
- **Hyperparameter Optimization** was done using **RandomSearchCV** to efficiently explore a large parameter space.
- **Cross-Validation** was applied to ensure robustness across different subsets of the data.

### **Feature Importance Analysis**
The most influential features in detecting fraudulent transactions included:
- **Total number of items in the basket**
- **Total price of the basket**
- **Certain high-risk product categories (e.g., electronics, high-value items)**
- **Brand and model interactions** (e.g., repeated purchases of high-end smartphones)

---

## 📌 Evaluation Metric: Precision-Recall AUC (PR-AUC)

Given the **imbalance in fraud cases (1.4%)**, accuracy is not an appropriate metric. Instead, we use **Precision-Recall AUC (PR-AUC)**, which provides a better assessment of model performance for the minority (fraud) class.

### PR-AUC Formula

PR-AUC = Σ (Recall_n - Recall_{n-1}) × Precision_n

where:

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

This is implemented in **scikit-learn’s** `average_precision_score`.






