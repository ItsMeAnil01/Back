from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:8080"]}})

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected input features
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Feature range validation
FEATURE_RANGES = {
    "age": (0, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (0, 300),
    "chol": (0, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (0, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3)
}

# Model version
MODEL_VERSION = "1.0.0"

# Load model, scaler, and selected features
try:
    if not os.path.exists('heart_attack_model.pkl') or not os.path.exists('scaler.pkl') or not os.path.exists('selected_features.pkl'):
        raise FileNotFoundError("Model, scaler, or selected features file not found. Run train_model.py first.")
    model = joblib.load('heart_attack_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    model_version = getattr(model, 'version', 'unknown')
    model_type = type(model).__name__
    logger.info(f"Model, scaler, and features loaded successfully. Model type: {model_type}, Version: {model_version}, Features: {selected_features}")
except Exception as e:
    logger.error(f"Failed to load model, scaler, or features: {str(e)}")
    raise

def validate_input(data: Dict[str, str]) -> tuple[pd.DataFrame, str]:
    if not all(feature in data for feature in EXPECTED_FEATURES):
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        return None, f"Missing required features: {', '.join(missing)}"
    input_data = []
    errors = []
    for feature in EXPECTED_FEATURES:
        value = data.get(feature)
        try:
            float_val = float(value)
            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= float_val <= max_val):
                errors.append(f"{feature} must be between {min_val} and {max_val}")
            input_data.append(float_val)
        except (ValueError, TypeError):
            errors.append(f"{feature} must be a numeric value")
    if errors:
        error_msg = "; ".join(errors)
        logger.warning(f"Input validation failed: {error_msg}")
        return None, error_msg
    input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
    # Select only the features used in training
    input_df = input_df[selected_features]
    return input_df, ""

@app.route('/', methods=['GET'])
def root():
    logger.info("Accessed root endpoint")
    return jsonify({'status': 'Health Oracle API is running', 'version': '1.0.0'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data received")
            return jsonify({'error': 'No input data provided'}), 400
        logger.info(f"Received input: {data}")
        input_df, error = validate_input(data)
        if error:
            logger.warning(f"Input validation failed: {error}")
            return jsonify({'error': error}), 400
        input_data_scaled = scaler.transform(input_df)
        prediction = model.predict_proba(input_data_scaled)[0][1] * 100
        result = {'Heart Attack Risk': f'{prediction:.2f}%'}
        logger.info(f"Prediction: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {str(e)}")
        raise