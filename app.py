from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import pandas as pd
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:8080"]}})

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected input features
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Numerical features
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Feature ranges for validation
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

# Define categories for feature validation
CATEGORICAL_CATEGORIES = {
    'sex': [0, 1],
    'cp': [0, 1, 2, 3],
    'fbs': [0, 1],
    'restecg': [0, 1, 2],
    'exang': [0, 1],
    'slope': [0, 1, 2],
    'ca': [0, 1, 2, 3, 4],
    'thal': [0, 1, 2, 3]
}

# Load model and preprocessor
model = None
preprocessor = None
try:
    if os.path.exists('heart_attack_model.pkl') and os.path.exists('preprocessor.pkl'):
        model = joblib.load('heart_attack_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        logger.info("Model and preprocessor loaded successfully")
    else:
        logger.warning("Model or preprocessor file not found. Predictions will fail until train_model.py is run.")
except Exception as e:
    logger.error(f"Failed to load model or preprocessor: {str(e)}")

# Calculate expected transformed feature count
EXPECTED_TRANSFORMED_FEATURES = (
        5 +  # Numerical features: age, trestbps, chol, thalach, oldpeak
        sum(len(CATEGORICAL_CATEGORIES[feat]) for feat in CATEGORICAL_CATEGORIES)  # Categorical features
)
logger.info(f"Expected transformed features: {EXPECTED_TRANSFORMED_FEATURES}")


@app.route('/', methods=['GET'])
def index():
    logger.info("Accessed root endpoint")
    return jsonify({"message": "Health Oracle API is running. Use /predict for predictions."})


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    logger.debug("Accessed favicon endpoint")
    return Response(status=204)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        logger.error("Model or preprocessor not loaded")
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503
    try:
        data = request.json
        logger.info(f"Received input: {data}")
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No input data provided'}), 400

        # Extract and validate features
        received_features = list(data.keys())
        logger.info(f"Received features: {received_features}")

        if set(received_features) != set(EXPECTED_FEATURES):
            extra = set(received_features) - set(EXPECTED_FEATURES)
            missing = set(EXPECTED_FEATURES) - set(received_features)
            error_msg = f"Invalid features. Extra: {extra}, Missing: {missing}"
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        input_data = {}
        errors = []
        for feature in EXPECTED_FEATURES:
            try:
                value = float(data[feature])
                min_val, max_val = FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    errors.append(f"{feature} must be between {min_val} and {max_val}")
                if feature in CATEGORICAL_CATEGORIES and value not in CATEGORICAL_CATEGORIES[feature]:
                    errors.append(f"{feature} must be one of {CATEGORICAL_CATEGORIES[feature]}")
                input_data[feature] = value
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a numeric value")

        if errors:
            error_msg = "; ".join(errors)
            logger.warning(f"Input validation failed: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Create input DataFrame
        input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
        logger.debug(f"Input DataFrame:\n{input_df}")

        # Transform and predict
        input_transformed = preprocessor.transform(input_df)
        logger.debug(f"Transformed input shape: {input_transformed.shape}")
        if input_transformed.shape[1] != EXPECTED_TRANSFORMED_FEATURES:
            error_msg = f"Transformed input has {input_transformed.shape[1]} features, expected {EXPECTED_TRANSFORMED_FEATURES}"
            logger.error(error_msg)
            return jsonify({'error': f"Internal server error: {error_msg}"}), 500

        # Log transformed feature names
        feature_names = (
                NUMERICAL_FEATURES +
                [f"{feat}_{cat}" for feat in CATEGORICAL_CATEGORIES
                 for cat in CATEGORICAL_CATEGORIES[feat]]
        )
        logger.debug(f"Transformed feature names: {feature_names}")

        prediction = model.predict_proba(input_transformed)[0]
        result = {'Heart Attack Risk': f'{prediction[1] * 100:.2f}%'}
        logger.info(f"Prediction: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    logger.info("Accessed health endpoint")
    return jsonify({'status': 'API is running', 'model_loaded': model is not None})


if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    logger.info("Starting Flask server")
    app.run(debug=True, host='127.0.0.1', port=5000)