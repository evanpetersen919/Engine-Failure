"""
Shared model and data loading for both Gradio and FastAPI
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """LSTM model for RUL prediction - matches training architecture"""
    def __init__(self, n_features, hidden_dim1=64, hidden_dim2=32, fc_dim=32, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(n_features, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_dim2, fc_dim)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(fc_dim, 1)
        
    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        lstm2_last = lstm2_out[:, -1, :]
        
        fc1_out = self.fc1(lstm2_last)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        
        output = self.fc2(fc1_out)
        
        return output

# -------------------------------------------------------------------------
# Global Variables (loaded once)
# -------------------------------------------------------------------------

model = None
X_test = None
y_test = None
metadata = None
shap_results = None

def load_model_and_data():
    """Load model and data once - called at startup"""
    global model, X_test, y_test, metadata, shap_results
    
    if model is not None:
        return model, X_test, y_test, None, metadata, shap_results  # Already loaded, return values
    
    # Load metadata first to get n_features
    with open('processed_data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    n_features = metadata['n_features']
    
    # Load model with correct architecture
    model = LSTMModel(n_features=n_features).to(device)
    checkpoint = torch.load('models/lstm_best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load SHAP results
    with open('shap_results/explainability_results.pkl', 'rb') as f:
        shap_results = pickle.load(f)
    
    print("✅ Model and data loaded successfully!")
    
    return model, X_test, y_test, None, metadata, shap_results

# -------------------------------------------------------------------------
# Prediction Functions
# -------------------------------------------------------------------------

def predict_rul(sequence_data):
    """
    Predict RUL for a given engine sequence
    
    Args:
        sequence_data: numpy array of shape (30, 14) - 30 timesteps, 14 sensors
    
    Returns:
        predicted_rul: float
        confidence: str
        top_features: list of feature names
        top_values: list of importance values
    """
    # Ensure model is loaded
    if model is None:
        load_model_and_data()
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0][0]
    
    # Calculate confidence (0-1 scale, higher for predictions in typical range)
    if 20 < prediction < 110:
        confidence = 0.95
    elif 10 < prediction < 120:
        confidence = 0.75
    else:
        confidence = 0.50
    
    # Get top contributing sensors from SHAP values
    # Convert dict to DataFrame if needed
    if isinstance(shap_results['global_importance'], dict):
        global_importance_df = pd.DataFrame(shap_results['global_importance'])
    else:
        global_importance_df = shap_results['global_importance']
    
    top_sensors_df = global_importance_df.head(5)[['Feature', 'Importance']]
    
    return float(prediction), confidence, top_sensors_df

def get_test_engine(engine_index):
    """Get test engine data by index"""
    if X_test is None:
        load_model_and_data()
    
    if engine_index < 0 or engine_index >= len(X_test):
        raise ValueError(f"Engine index must be between 0 and {len(X_test)-1}")
    
    return X_test[engine_index], y_test[engine_index]

def get_model_info():
    """Get model metadata"""
    if metadata is None:
        load_model_and_data()
    
    return {
        'model_type': 'LSTM',
        'architecture': '2-layer LSTM (64→32 hidden units)',
        'input_shape': '(30, 14)',
        'parameters': '~45K',
        'test_rmse': 14.40,
        'test_mae': 10.80,
        'test_r2': 0.765,
        'dataset': 'NASA C-MAPSS FD001',
        'num_test_engines': len(X_test) if X_test is not None else 100
    }
