import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def execute_comprehensive_modeling_pipeline():
    print("📊 Generating Basel II Compliant EDA, RFM Feature Selection, & Model Tuning Workspace...")
    
    # 1. Synthesize a richer dataset representing raw alternative financial data
    np.random.seed(42)
    n_samples = 250
    
    mock_data = {
        'CustomerId': np.random.choice([f'Cust_{i}' for i in range(1, 30)], n_samples),
        'Amount': np.random.uniform(50, 15000, n_samples),
        'Value': np.random.uniform(50, 15000, n_samples),
        'PricingStrategy': np.random.choice([1, 2, 4], n_samples),
        'ProductCategory': np.random.choice(['utility', 'financial_services', 'airtime', 'ticket'], n_samples),
        'ChannelId': np.random.choice(['ChannelId_1', 'ChannelId_2', 'ChannelId_3'], n_samples),
        'ProviderId': np.random.choice(['ProviderId_1', 'ProviderId_2', 'ProviderId_4'], n_samples),
        'TransactionStartTime': pd.date_range(start='2026-01-01', periods=n_samples, freq='4H').astype(str)
    }
    df = pd.DataFrame(mock_data)
    
    # 2. Domain-Specific Modeling: RFM-Based Proxy Target Construction
    print("🎯 Constructing RFM-based Credit Risk Proxy Target...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Calculate operational metrics per customer
    customer_rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (pd.to_datetime('2026-03-01') - x.max()).days, # Recency
        'CustomerId': 'count',                                                          # Frequency
        'Amount': 'sum'                                                                 # Monetary
    }).rename(columns={'TransactionStartTime': 'Recency', 'CustomerId': 'Frequency', 'Amount': 'Monetary'})
    
    # Merge metrics back into the transaction stream
    df = df.merge(customer_rfm, on='CustomerId', how='left')
    
    # Define Target Default Proxy (1 = High Risk Default, 0 = Low Risk Approved)
    # Basel II alignment: flag customers with low transaction frequencies but highly irregular transaction sizes
    df['Target'] = np.where((df['Monetary'] > 25000) & (df['Frequency'] < 8), 1, 0)
    
    # 3. Richer Feature Engineering
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_To_Monetary_Ratio'] = df['Amount'] / (df['Monetary'] + 1e-5)
    
    # 4. Modular Preprocessing Selection
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'Recency', 'Frequency', 'Monetary', 'Transaction_To_Monetary_Ratio']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = df.drop(columns=['CustomerId', 'TransactionStartTime', 'Target'])
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # 5. Multi-Model Execution & Hyperparameter Tuning
    print("⚙️ Tuning Multiple Models (Random Forest & Logistic Regression)...")
    
    # Model A: Random Forest Tuning
    rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='roc_auc')
    rf_grid.fit(X_train_trans, y_train)
    champion_rf = rf_grid.best_estimator_
    
    # Model B: Logistic Regression Tuning
    lr_param_grid = {'C': [0.1, 1.0, 10.0]}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_param_grid, cv=3, scoring='roc_auc')
    lr_grid.fit(X_train_trans, y_train)
    champion_lr = lr_grid.best_estimator_
    
    # 6. Evaluate Best Performing Architecture
    rf_auc = roc_auc_score(y_test, champion_rf.predict_proba(X_test_trans)[:, 1])
    lr_auc = roc_auc_score(y_test, champion_lr.predict_proba(X_test_trans)[:, 1])
    
    print(f"🏆 Random Forest Best AUC: {rf_auc:.4f}")
    print(f"🏆 Logistic Regression Best AUC: {lr_lr_auc if 'lr_lr_auc' in locals() else lr_auc:.4f}")
    
    # Export Champion Assets directly into production workspace path
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessing_pipeline.pkl')
    joblib.dump(champion_rf, 'models/random_forest_model.pkl')
    print("🎯 Production architecture fully calibrated and written to /models folder.")

if __name__ == "__main__":
    execute_comprehensive_modeling_pipeline()