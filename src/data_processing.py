import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    """
    Transforms raw transaction data by extracting temporal components
    and structuring preprocessing transformers.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Year'] = df['TransactionStartTime'].dt.year
    
    numeric_features = ['Amount', 'Value', 'PricingStrategy']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor, df