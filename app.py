from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import joblib

app = Flask(__name__)

# Define the model architecture
class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CreditRiskNN(input_dim=23)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict credit default probability"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = np.array([[
            float(data['limit_bal']),
            int(data['sex']),
            int(data['education']),
            int(data['marriage']),
            int(data['age']),
            int(data['pay_0']),
            int(data['pay_2']),
            int(data['pay_3']),
            int(data['pay_4']),
            int(data['pay_5']),
            int(data['pay_6']),
            float(data['bill_amt1']),
            float(data['bill_amt2']),
            float(data['bill_amt3']),
            float(data['bill_amt4']),
            float(data['bill_amt5']),
            float(data['bill_amt6']),
            float(data['pay_amt1']),
            float(data['pay_amt2']),
            float(data['pay_amt3']),
            float(data['pay_amt4']),
            float(data['pay_amt5']),
            float(data['pay_amt6'])
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            probability = model(features_tensor).item()
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High Risk"
            recommendation = "Deny Credit Application"
        elif probability > 0.5:
            risk_level = "Medium Risk"
            recommendation = "Review Manually or Offer Lower Limit"
        else:
            risk_level = "Low Risk"
            recommendation = "Approve Credit Application"
        
        return jsonify({
            'success': True,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)