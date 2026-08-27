# customer_churn_analysis
Customer Churn Analysis - Data and BI Project using Python, ML and Power BI



#Source code

"""
Customer Churn Analysis - Complete Source Code
Project: Customer Churn Analysis
Dataset: IBM Telco Churn (7043 records, 21 columns)
Author: Lakshmilavanya
GitHub: https://github.com/Lavanya4207/customer-churn-analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle

# 1. DATA LOADING
print("Loading Dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Original Shape: {df.shape}")  # (7043, 21)

# 2. DATA CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop_duplicates(inplace=True)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# 3. FEATURE ENGINEERING
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,100], labels=['0-12','12-24','24-48','48+'])
df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x!='Month-to-month' else 0)
df['IsElectronicCheck'] = df['PaymentMethod'].apply(lambda x: 1 if x=='Electronic check' else 0)
df['IsHighMonthlyCharge'] = df['MonthlyCharges'].apply(lambda x: 1 if x>70 else 0)

# 4. PREPROCESSING & SPLIT
X = df.drop(['customerID','Churn'], axis=1)
y = df['Churn']

cat_features = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','TenureGroup']
num_features = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','IsLongTermContract',
                'IsElectronicCheck','IsHighMonthlyCharge']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.20, random_state=42, stratify=y)

# SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Train: {X_train_bal.shape}, Churn Ratio: {y_train_bal.mean():.2%}")

# 5. MODEL TRAINING - RANDOM FOREST (Main Model)
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_bal, y_train_bal)

# 6. EVALUATION
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

print(f"\n=== MODEL RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  # 84-87%
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")  # 88-92%
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_names = preprocessor.get_feature_names_out()
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))

# 7. SAVE MODEL
import os
os.makedirs('models', exist_ok=True)
with open('models/churn_model.pkl','wb') as f:
    pickle.dump(rf_model, f)
with open('models/preprocessor.pkl','wb') as f:
    pickle.dump(preprocessor, f)

# 8. PREDICTION FOR ACTIVE CUSTOMERS
df_active = df[df['Churn']==0].copy()
X_active_raw = df_active.drop(['customerID','Churn'], axis=1)
X_active_processed = preprocessor.transform(X_active_raw)
df_active['ChurnProbability'] = rf_model.predict_proba(X_active_processed)[:,1]

def risk_level(prob):
    if prob < 0.40:
        return 'Low Risk'
    elif prob < 0.70:
        return 'Medium Risk'
    else:
        return 'High Risk'

df_active['RiskLevel'] = df_active['ChurnProbability'].apply(risk_level)
high_risk = df_active[df_active['ChurnProbability']>=0.70]
print(f"\nHigh Risk Customers: {len(high_risk)}")

high_risk[['customerID','tenure','Contract','MonthlyCharges','ChurnProbability','RiskLevel']].to_csv('high_risk_customers.csv', index=False)
print("High risk list saved - Ready for Power BI Dashboard!")
print("\n=== PROJECT COMPLETED SUCCESSFULLY ===")
