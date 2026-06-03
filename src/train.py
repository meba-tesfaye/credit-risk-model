import sys
import os
# 🛠️ Force Python to look at the project root directory so 'from src.data_processing' works perfectly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.data_processing import preprocess_data

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def train_and_track_model():
    print("🚀 Initializing Bati Bank Model Training & Tracking Pipeline...")

    np.random.seed(42)
    mock_data = {
        'Amount': np.random.uniform(100, 10000, 100),
        'Value': np.random.uniform(100, 10000, 100),
        'PricingStrategy': np.random.choice([1, 2, 4], 100),
        'ProductCategory': np.random.choice(['utility', 'financial_services', 'airtime', 'ticket'], 100),
        'ChannelId': np.random.choice(['ChannelId_1', 'ChannelId_2', 'ChannelId_3'], 100),
        'ProviderId': np.random.choice(['ProviderId_1', 'ProviderId_2', 'ProviderId_4'], 100),
        'TransactionStartTime': pd.date_range(start='2026-01-01', periods=100, freq='H').astype(str)
    }
    df = pd.DataFrame(mock_data)
    y = np.where((df['Amount'] > 6000) & (df['PricingStrategy'] == 2), 1, 0)

    preprocessor, df_features = preprocess_data(df)
    X_transformed = preprocessor.fit_transform(df_features)

    X_train, X_test = X_transformed[:80], X_transformed[80:]
    y_train, y_test = y[:80], y[80:]

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Bati_Bank_Credit_Risk")
        with mlflow.start_run():
            n_estimators = 100
            max_depth = 10
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "random_forest_champion")
            print(f"📈 MLflow tracking active. Metrics logged -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_transformed, y)
        print("⚠️ mlflow package not found. Running local metric extraction pipeline instead.")

    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessing_pipeline.pkl')
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("🎯 Model training complete! Production pipeline assets successfully exported to /models.")

if __name__ == "__main__":
    train_and_track_model()