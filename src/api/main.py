from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Bati Bank Credit Scoring Engine")

class TransactionPayload(BaseModel):
    Amount: float = Field(..., description="Transaction amount in local currency")
    Value: float = Field(..., description="Gross value field parameter")
    PricingStrategy: int = Field(..., description="Category index for pricing structure")
    ProductCategory: str = Field(..., description="Categorical label for product type")
    ChannelId: str = Field(..., description="Platform deployment interface identifier")
    ProviderId: str = Field(..., description="Sourcing provider identifier")
    TransactionStartTime: str = Field(..., description="ISO formatted time string")

@app.post("/predict")
def evaluate_credit_risk(payload: TransactionPayload):
    # 🛠️ Defensive Programming & Strict Error Handling
    try:
        # Validate baseline logical boundaries to catch ingestion anomalies
        if payload.Amount < 0 or payload.Value < 0:
            raise HTTPException(
                status_code=400, 
                detail="Financial input violation: 'Amount' and 'Value' parameters must be non-negative numeric metrics."
            )
            
        if payload.PricingStrategy not in [1, 2, 4]:
            raise HTTPException(
                status_code=400,
                detail=f"Categorical boundary overflow: Received invalid PricingStrategy [{payload.PricingStrategy}]. Authorized options are [1, 2, 4]."
            )

        # Mock simulation for model inference array execution
        # (Replace with model.predict logic cleanly)
        is_high_risk = 1 if (payload.Amount > 6000 and payload.PricingStrategy == 2) else 0
        mock_pd = 0.8642 if is_high_risk else 0.0315
        
        return {
            "status": "success",
            "credit_evaluation": {
                "risk_prediction": is_high_risk,
                "risk_description": "High Risk Default Profile" if is_high_risk else "Approved Credit Baseline",
                "probability_of_default": mock_pd
            }
        }
        
    except HTTPException as http_err:
        raise http_err
    except Exception as general_err:
        # Prevent generic 500 runtime stack-trace leaks to the frontend
        raise HTTPException(
            status_code=500,
            detail=f"Internal model pipeline parsing runtime exception: {str(general_err)}"
        )