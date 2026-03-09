import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Scale 'Amount' and 'Time'
    # RobustScaler is less prone to outliers than StandardScaler
    scaler = RobustScaler()
    
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop the original unscaled columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Define Features (X) and Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split into Train/Test sets (80/20)
    # stratify=y is KEY here to ensure both sets have the same % of fraud
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_clean_data('creditcard.csv')
    print(f"Training shapes: {X_train.shape}, {y_train.shape}")
    print(f"Fraud count in training set: {y_train.sum()}")