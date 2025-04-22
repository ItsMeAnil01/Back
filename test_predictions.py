import requests
import json
import logging

# Set up logging
logging.basicConfig(
    filename='test_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Backend API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Test cases
TEST_CASES = {
    "Default Values": {
        "age": 45,
        "sex": 1,
        "cp": 0,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0,
        "slope": 0,
        "ca": 0,
        "thal": 0
    },
    "Set 1 (Typical Case)": {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 132,
        "chol": 353,
        "fbs": 0,
        "restecg": 0,
        "thalach": 132,
        "exang": 1,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 1,
        "thal": 2
    },
    "Set 2 (High-Risk Case)": {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 150,
        "chol": 407,
        "fbs": 1,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 2.4,
        "slope": 2,
        "ca": 3,
        "thal": 2
    }
}

def test_prediction(test_name: str, data: dict):
    """Submit test case to API and log result."""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info(f"{test_name} Prediction: {result}")
        print(f"{test_name} Prediction: {result}")
        return result
    except requests.RequestException as e:
        logger.error(f"{test_name} failed: {str(e)}")
        print(f"{test_name} failed: {str(e)}")
        return None

def main():
    print("Starting automated prediction tests")
    logger.info("Starting automated prediction tests")
    for test_name, data in TEST_CASES.items():
        test_prediction(test_name, data)
    print("Completed automated prediction tests")
    logger.info("Completed automated prediction tests")

if __name__ == '__main__':
    main()