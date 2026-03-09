from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

def apply_smote(X_train, y_train):
    print(f"Original dataset shape: {Counter(y_train)}")
    
    # Initialize SMOTE
    # sampling_strategy=0.1 means we want the minority class to be 10% of the majority
    sm = SMOTE(sampling_strategy=0.1, random_state=42)
    
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    print(f"Resampled dataset shape: {Counter(y_res)}")
    return X_res, y_res

# Quick test logic
if __name__ == "__main__":
    from preprocess import load_and_clean_data
    X_train, X_test, y_train, y_test = load_and_clean_data('creditcard.csv')
    X_res, y_res = apply_smote(X_train, y_train)