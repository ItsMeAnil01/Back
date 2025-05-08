import requests
import logging
import json

# Set up logging
logging.basicConfig(
    filename='test_predictions.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prediction(data, name):
    try:
        logger.debug(f"Sending data for {name}: {json.dumps(data, indent=2)}")
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        logger.debug(f"Response status for {name}: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"{name}: {result}")
            logger.info(f"{name}: {result}")
        else:
            error_text = response.text
            print(f"{name} failed: {response.status_code} {error_text}")
            logger.error(f"{name} failed: {response.status_code} {error_text}")
    except Exception as e:
        print(f"{name} failed: {str(e)}")
        logger.error(f"{name} failed: {str(e)}")

def main():
    print("Starting automated prediction tests")
    logger.info("Starting automated prediction tests")

    # Test cases
    default_values = {
        "age": 45, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0,
        "slope": 0, "ca": 0, "thal": 0
    }

    typical_case = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1, "oldpeak": 1.5,
        "slope": 1, "ca": 1, "thal": 2
    }

    high_risk_case = {
        "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 300,
        "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.5,
        "slope": 2, "ca": 3, "thal": 3
    }

    test_prediction(default_values, "Default Values")
    test_prediction(typical_case, "Set 1 (Typical Case)")
    test_prediction(high_risk_case, "Set 2 (High-Risk Case)")

    print("Completed automated prediction tests")
    logger.info("Completed automated prediction tests")

if __name__ == '__main__':
    main()