
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
lr_model = joblib.load("linear_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "House Price Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Please provide input in JSON format with a "features" list.'})
    
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)

    lr_pred = lr_model.predict(features_scaled)[0]
    xgb_pred = xgb_model.predict(features_scaled)[0]

    return jsonify({
        'LinearRegression_Prediction': round(float(lr_pred), 2),
        'XGBoost_Prediction': round(float(xgb_pred), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
