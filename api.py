"""
FastAPI REST API for Engine Failure Prediction
Provides programmatic access to the LSTM RUL prediction model
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from model_loader import load_model_and_data, predict_rul, get_test_engine, get_model_info

# -------------------------------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------------------------------

app = FastAPI(
    title="Aircraft Engine RUL Prediction API",
    description="REST API for predicting Remaining Useful Life of aircraft engines using LSTM + SHAP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    load_model_and_data()

# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------

class SensorSequence(BaseModel):
    """Sensor readings for 30 timesteps"""
    sequence: List[List[float]] = Field(
        ..., 
        description="2D array of shape (30, 14) - 30 timesteps, 14 sensors",
        example=[[0.0]*14]*30
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]] * 30
            }
        }

class PredictionResponse(BaseModel):
    """RUL prediction response"""
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    confidence: str = Field(..., description="Confidence level: High, Medium, or Low")
    top_features: List[str] = Field(..., description="Top 5 most important sensor features")
    top_values: List[float] = Field(..., description="Importance scores for top features")
    status: str = Field(..., description="Engine status: HEALTHY, WARNING, or CRITICAL")
    recommendation: str = Field(..., description="Maintenance recommendation")

class BatchPredictionRequest(BaseModel):
    """Multiple engine sequences for batch prediction"""
    sequences: List[List[List[float]]] = Field(
        ...,
        description="List of sequences, each of shape (30, 14)"
    )

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    count: int

class TestEngineRequest(BaseModel):
    """Request prediction for a test engine by index"""
    engine_index: int = Field(..., ge=0, lt=100, description="Test engine index (0-99)")

class ModelInfoResponse(BaseModel):
    """Model metadata and performance metrics"""
    model_type: str
    architecture: str
    input_shape: str
    parameters: str
    test_rmse: float
    test_mae: float
    test_r2: float
    dataset: str
    num_test_engines: int

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def get_status_and_recommendation(rul: float) -> tuple[str, str]:
    """Determine engine status and maintenance recommendation"""
    if rul < 30:
        status = "CRITICAL"
        recommendation = "⚠️ IMMEDIATE ACTION REQUIRED: Schedule emergency maintenance. Engine should be grounded until inspection."
    elif rul < 70:
        status = "WARNING"
        recommendation = "⚡ ATTENTION NEEDED: Plan maintenance within next flight cycle. Increased monitoring recommended."
    else:
        status = "HEALTHY"
        recommendation = "✅ NORMAL OPERATION: Continue routine monitoring. Next inspection as scheduled."
    
    return status, recommendation

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """API root - returns basic info"""
    return {
        "message": "Aircraft Engine RUL Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "test_engine": "/test_engine/{engine_index}",
            "model_info": "/model_info"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics"""
    try:
        info = get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: SensorSequence):
    """
    Predict RUL for a single engine sequence
    
    Requires a sensor sequence of shape (30, 14):
    - 30 timesteps
    - 14 sensor readings per timestep
    
    Returns predicted RUL, confidence, top features, and maintenance recommendation
    """
    try:
        # Validate input shape
        sequence = np.array(request.sequence)
        if sequence.shape != (30, 14):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence shape {sequence.shape}. Expected (30, 14)"
            )
        
        # Make prediction
        predicted_rul, confidence, top_features, top_values = predict_rul(sequence)
        
        # Get status and recommendation
        status, recommendation = get_status_and_recommendation(predicted_rul)
        
        return PredictionResponse(
            predicted_rul=predicted_rul,
            confidence=confidence,
            top_features=top_features,
            top_values=top_values,
            status=status,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict RUL for multiple engine sequences
    
    Accepts a list of sequences, each of shape (30, 14)
    Returns predictions for all sequences
    """
    try:
        predictions = []
        
        for i, seq in enumerate(request.sequences):
            sequence = np.array(seq)
            if sequence.shape != (30, 14):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid sequence shape at index {i}: {sequence.shape}. Expected (30, 14)"
                )
            
            # Make prediction
            predicted_rul, confidence, top_features, top_values = predict_rul(sequence)
            status, recommendation = get_status_and_recommendation(predicted_rul)
            
            predictions.append(PredictionResponse(
                predicted_rul=predicted_rul,
                confidence=confidence,
                top_features=top_features,
                top_values=top_values,
                status=status,
                recommendation=recommendation
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/test_engine/{engine_index}", response_model=PredictionResponse, tags=["Prediction"])
async def predict_test_engine(engine_index: int):
    """
    Predict RUL for a test engine by index (0-99)
    
    Useful for testing with real NASA C-MAPSS test data
    """
    try:
        # Get test engine data
        sequence, actual_rul = get_test_engine(engine_index)
        
        # Make prediction
        predicted_rul, confidence, top_features, top_values = predict_rul(sequence)
        status, recommendation = get_status_and_recommendation(predicted_rul)
        
        # Add actual RUL to recommendation for test engines
        error = abs(predicted_rul - actual_rul)
        recommendation += f" | Actual RUL: {actual_rul:.1f} cycles | Error: {error:.1f} cycles"
        
        return PredictionResponse(
            predicted_rul=predicted_rul,
            confidence=confidence,
            top_features=top_features,
            top_values=top_values,
            status=status,
            recommendation=recommendation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------------------------------------------------
# Run Server
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
