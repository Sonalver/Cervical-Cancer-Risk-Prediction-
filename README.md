# Cervical-Cancer-Risk-Prediction-
Cervical cancer is a type of cancer that develops in the cervix, the lower part of the uterus that connects to the vagina. Cervical cancer prediction using XG-BOOST 
Machine Learning Project using XGBoost, Data Cleaning, EDA, and Visualization
This project builds a Cervical Cancer Risk Prediction Model using machine learning techniques.
The dataset contains medical, sexual health, and diagnostic features that help identify whether a patient is at risk of cervical cancer.

We perform:
Data cleaning & preprocessing
Handling missing values
Exploratory Data Analysis (EDA)
Correlation analysis
Feature scaling
Model training using XGBoost Classifier
Evaluation using accuracy, confusion matrix, classification report
Visualization using Matplotlib, Seaborn

Dataset Details
Name: Cervical Cancer Risk Factors Dataset
Source: UCI Machine Learning Repository
Rows: ~858
Columns: 36 (after cleaning: 34)

Target Variable:
Biopsy → 1 (positive for cervical cancer), 0 (negative)

Data Preprocessing Steps
✔ Replace ? with NaN
Many features contained missing values represented as "?".

✔ Drop columns with >80% missing values
STDs: Time since first diagnosis
STDs: Time since last diagnosis

✔ Convert all features to numeric
✔ Fill missing values using column-wise mean
✔ Scale data using StandardScaler
✔ Train-test split (75% training, 25% testing)

Exploratory Data Analysis (EDA)
1. Missing Value Heatmap
Helps visualize how many fields in the dataset are missing.

2. Correlation Matrix
Understanding feature relationships.

3. Histograms
Distribution of all numerical features.

4. Confusion Matrix
Evaluate classification performance.

Models Used
Model 1
XGBClassifier(
    learning_rate = 0.1,
    max_depth = 5,
    n_estimators = 10
)

Model 2 (Improved)
XGBClassifier(
    learning_rate = 0.1,
    max_depth = 50,
    n_estimators = 100
)

Model Evaluation
Classification Metrics
Accuracy
Precision
Recall
F1-score

Key Results
Model 2 performs significantly better with deeper trees and higher estimators.
Feature scaling improves model stability.

Dataset is highly imbalanced, so further improvements may be needed:
SMOTE oversampling
Hyperparameter tuning
Cross-validation
