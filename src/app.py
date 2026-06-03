from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Bati Bank Credit Risk Assessment API",
    description="Real-time transaction risk scoring using a champion Random Forest model.",
    version="1.0.0"
)

# Define absolute paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'preprocessing_pipeline.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')

# Load the preprocessing pipeline and model into memory
try:
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    print("🎯 Preprocessing pipeline and Random Forest model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    pipeline = None
    model = None

# Define structural schema for incoming requests
class TransactionInput(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: int
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    TransactionStartTime: str 

@app.get("/")
def home():
    return {"status": "healthy", "message": "Credit Risk API is fully operational"}

@app.post("/predict")
def predict_risk(input_data: TransactionInput):
    if pipeline is None or model is None:
        raise HTTPException(status_code=500, detail="Model assets are unavailable on the server.")
    
    try:
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # 🛠️ Extract temporal pipeline features dynamically from the timestamp string
        input_df['TransactionStartTime'] = pd.to_datetime(input_df['TransactionStartTime'])
        input_df['Hour'] = input_df['TransactionStartTime'].dt.hour
        input_df['Day'] = input_df['TransactionStartTime'].dt.day
        input_df['Month'] = input_df['TransactionStartTime'].dt.month
        input_df['Year'] = input_df['TransactionStartTime'].dt.year
        
        # Pass the fully reconstructed DataFrame through transformation and inference
        X_transformed = pipeline.transform(input_df)
        prediction = int(model.predict(X_transformed)[0])
        probability = float(model.predict_proba(X_transformed)[0][1])
        
        risk_status = "High Risk (Potential Default)" if prediction == 1 else "Low Risk (Approved)"
        
        return {
            "prediction": prediction,
            "risk_score_probability": round(probability, 4),
            "assessment": risk_status
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution failure: {str(e)}")