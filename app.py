import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model
model = joblib.load('fraud_model.pkl')

st.title("🛡️ Financial Fraud Detection System")
st.subheader("Real-time Anomaly Detection using XGBoost & SHAP")

st.write("""
This system analyzes credit card transactions to identify potential fraudulent activity.
Enter the transaction details below to get a risk assessment.
""")

# Create Input Fields
# In a real scenario, these would be V1-V28
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    time = st.number_input("Seconds since last transaction", min_value=0, value=3600)

with col2:
    # Simulating the V-features for the sake of the demo
    v1 = st.slider("Feature V1 (Anonymized)", -5.0, 5.0, 0.0)
    v2 = st.slider("Feature V2 (Anonymized)", -5.0, 5.0, 0.0)

# Prediction Logic
if st.button("Analyze Transaction"):
    # Prepare the input (filling other V-features with 0)
    features = np.zeros(30)
    features[28] = amount # Scaled amount index
    features[29] = time   # Scaled time index
    features[0] = v1
    features[1] = v2
    
    prediction = model.predict(features.reshape(1, -1))
    probability = model.predict_proba(features.reshape(1, -1))[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ HIGH RISK: Potential Fraud Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ LOW RISK: Transaction appears legitimate. (Probability: {probability:.2%})")

st.divider()
st.info("Note: This model was trained on a highly imbalanced dataset using SMOTE to improve recall for rare fraud events.")