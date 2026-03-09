import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('creditcard.csv')

# Check the class imbalance
print("--- Class Distribution ---")
print(df['Class'].value_counts())
print("\nPercentage of Fraud:")
print(df['Class'].value_counts(normalize=True) * 100)

# Visualize the Imbalance
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Transaction Class Distribution (0: Legit, 1: Fraud)')
plt.yscale('log') # log scale because Fraud is so small it won't show up otherwise
plt.show()

# Analyze Transaction Amounts
print("\n--- Transaction Amount Statistics ---")
print("Legit Transactions:")
print(df[df['Class'] == 0]['Amount'].describe())
print("\nFraudulent Transactions:")
print(df[df['Class'] == 1]['Amount'].describe())

# Amount Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount: Legit vs Fraud')
plt.ylim(0, 500)
plt.show()