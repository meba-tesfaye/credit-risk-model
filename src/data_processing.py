import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    """
    Transforms raw alternative transaction data by extracting temporal components,
    engineering domain-specific financial features, and structuring robust preprocessing pipelines.
    """
    # 🛠️ Extract rich temporal attributes dynamically
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Year'] = df['TransactionStartTime'].dt.year
    
    # 🛠️ Build baseline placeholders for RFM metrics if they aren't generated in the live API block
    if 'Recency' not in df.columns:
        df['Recency'] = 0.0
    if 'Frequency' not in df.columns:
        df['Frequency'] = 1.0
    if 'Monetary' not in df.columns:
        df['Monetary'] = df['Amount']
        
    # 🛠️ Feature Engineering: Financial ratio expansion
    df['Transaction_To_Monetary_Ratio'] = df['Amount'] / (df['Monetary'] + 1e-5)
    
    # Isolate feature profiles for column transformers
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 'Recency', 'Frequency', 'Monetary', 'Transaction_To_Monetary_Ratio']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor, df