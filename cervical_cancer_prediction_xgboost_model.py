# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb

# 2. Load Dataset
cancer_df = pd.read_csv('../../resources/cervical_cancer.csv')

print("\nFIRST 20 ROWS:")
print(cancer_df.head(20))

print("\nLAST 20 ROWS:")
print(cancer_df.tail(20))

# 3. Basic Analysis
print("\nDataset Info:")
cancer_df.info()

print("\nDataset Summary:")
print(cancer_df.describe())

# The dataset contains '?' as missing values → replace them
cancer_df = cancer_df.replace('?', np.nan)

# 4. Missing Value Heatmap
plt.figure(figsize=(18, 18))
sns.heatmap(cancer_df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.show()

print("\nMissing Value Count:")
print(cancer_df.isnull().sum())

# 5. Drop columns with >80% missing data
cols_to_drop = [
    'STDs: Time since first diagnosis',
    'STDs: Time since last diagnosis'
]

cancer_df.drop(columns=cols_to_drop, inplace=True)

# 6. Convert all columns to numeric
cancer_df = cancer_df.apply(pd.to_numeric)

# Fill missing values with mean
cancer_df = cancer_df.fillna(cancer_df.mean())

# 7. Confirm All Missing Values Resolved
plt.figure(figsize=(18, 18))
sns.heatmap(cancer_df.isnull(), yticklabels=False, cbar=False)
plt.title("Heatmap After Imputation")
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(25, 25))
sns.heatmap(cancer_df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 9. Histogram of All Numerical Features
cancer_df.hist(bins=10, figsize=(25, 25))
plt.show()

# 10. Prepare Data for Model
X = cancer_df.drop(columns=['Biopsy'])
y = cancer_df['Biopsy']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 11. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 12. Train XGBoost Model (Model 1)
model1 = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=10,
    eval_metric='logloss'
)

model1.fit(X_train, y_train)

print("\nMODEL 1 RESULTS")
print("Train Accuracy:", model1.score(X_train, y_train))
print("Test Accuracy:", model1.score(X_test, y_test))

y_pred1 = model1.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred1))

cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm1, annot=True, fmt='d')
plt.title("Confusion Matrix (Model 1)")
plt.show()

# 13. Train Improved XGBoost Model (Model 2)
model2 = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=50,
    n_estimators=100,
    eval_metric='logloss'
)

model2.fit(X_train, y_train)

print("\nMODEL 2 RESULTS")
print("Train Accuracy:", model2.score(X_train, y_train))
print("Test Accuracy:", model2.score(X_test, y_test))

y_pred2 = model2.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred2))

cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm2, annot=True, fmt='d')
plt.title("Confusion Matrix (Model 2)")
plt.show()
