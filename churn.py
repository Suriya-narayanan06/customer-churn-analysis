
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#  Dataset elco-Customer-Churn.csv

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully ")
print(df.head())
print(df.shape)


# Data Cleaning

# Drop customerID 
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Encode target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)
# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Contract vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# Encode Categorical Variables

df = pd.get_dummies(df, drop_first=True)

print("\nDataset shape after encoding:")
print(df.shape)

# Feature Selection

X = df.drop('Churn', axis=1)
y = df['Churn']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
