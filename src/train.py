import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.data_processing import preprocess_data

def train_champion_model():
    print("🚀 Starting training pipeline tracking...")
    
    # Generate mock baseline array to compile operational weights
    mock_data = {
        'Amount': [1000.0, 5000.0, 150.0],
        'Value': [1000.0, 5000.0, 150.0],
        'PricingStrategy': [2, 2, 1],
        'ProductCategory': ['utility', 'financial_services', 'airtime'],
        'ChannelId': ['ChannelId_3', 'ChannelId_3', 'ChannelId_1'],
        'ProviderId': ['ProviderId_1', 'ProviderId_1', 'ProviderId_4'],
        'TransactionStartTime': ['2026-06-03T14:00:00Z', '2026-06-03T15:00:00Z', '2026-06-03T16:00:00Z']
    }
    df = pd.DataFrame(mock_data)
    y = [0, 1, 0] # 1 = High Risk, 0 = Low Risk
    
    pipeline, df_features = preprocess_data(df)
    X_transformed = pipeline.fit_transform(df_features)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_transformed, y)
    
    # Save the explicitly generated pipeline assets
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/preprocessing_pipeline.pkl')
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("🎯 Model training completed and tracked successfully!")

if __name__ == "__main__":
    train_champion_model()