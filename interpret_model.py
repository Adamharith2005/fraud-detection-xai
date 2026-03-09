import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def interpret_fraud_model():
    # Load the model and some sample data
    model = joblib.load('fraud_model.pkl')
    # Small sample because SHAP can be computationally heavy
    df = pd.read_csv('creditcard.csv').sample(1000, random_state=42)
    X = df.drop('Class', axis=1)

    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('shap_summary.png')
    plt.show()

if __name__ == "__main__":
    interpret_fraud_model()