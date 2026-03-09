# 🛡️ Explainable AI for Fraud Detection
**An end-to-end system to identify financial anomalies using XGBoost, SMOTE and SHAP.**

### 📌 Project Overview
In financial datasets, fraud is often less than 0.2% of all transactions. This project demonstrates how to handle **extreme class imbalance** to build a robust detection system that prioritizes catching fraud (Recall) without overwhelming users with false alarms (Precision).

### Dataset: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv)

### Key Features:
* Imbalance Handling: Used SMOTE to synthetically balance the training set.
* Explainable AI (XAI): Integrated SHAP values to break the "black box" and explain why a transaction was flagged.
* Interactive Dashboard: A Streamlit UI for real-time risk assessment.

### 🛠️ Tech Stack
* Language: Python 3.10+
* Libraries: Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn
* Interpretability: SHAP (Shapley Additive Explanations)
* Deployment: Streamlit

### 📊 Results & Evaluation
Standard accuracy is misleading here (99.9% is easy). Instead, I focused on:
* AUPRC (Area Under Precision-Recall Curve): 0.8626
* Recall: 86% (How many frauds we actually caught)
* F1-Score: 0.77

This project successfully demonstrates a high-performance End-to-End Fraud Detection System built on real-world data constraints. By leveraging XGBoost and SMOTE, the model effectively navigates extreme class imbalance to achieve an AUPRC of 0.86 and an 86% Recall rate.

### Future Improvements
+ Implementing unsupervised anomaly detection
+ MLOps & API deployment

## 🚀 How to Run Locally
1. Clone the repo: https://github.com/Adamharith2005/fraud-detection-xai.git
2. Install dependencies: pip install -r requirements.txt
3. Run the dashboard: streamlit run app.py