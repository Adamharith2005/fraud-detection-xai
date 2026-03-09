import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score
import joblib # To save model

def train_fraud_model(X_train, y_train, X_test, y_test):
    # Initialize XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"AUPRC Score: {average_precision_score(y_test, y_probs):.4f}")

    joblib.dump(model, 'fraud_model.pkl')
    print("\nModel saved as fraud_model.pkl")
    
    return model

if __name__ == "__main__":
    from preprocess import load_and_clean_data
    X_train, X_test, y_train, y_test = load_and_clean_data('creditcard.csv')
    X_res, y_res = apply_smote(X_train, y_train)
    train_fraud_model(X_res, y_res, X_test, y_test)