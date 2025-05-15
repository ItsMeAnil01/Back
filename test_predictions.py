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

def test_prediction(data, name, expected_risk):
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
            actual_risk = float(result['Heart Attack Risk'].strip('%'))
            discrepancy = abs(actual_risk - expected_risk)
            status = "OK" if discrepancy < 10 else "WARNING"
            print(f"{name}: Actual: {result['Heart Attack Risk']}, Expected: {expected_risk:.2f}%, Discrepancy: {discrepancy:.2f}% ({status})")
            logger.info(f"{name}: Actual: {result['Heart Attack Risk']}, Expected: {expected_risk:.2f}%, Discrepancy: {discrepancy:.2f}% ({status})")
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

    # 25 diverse test cases with expected risk percentages
    test_cases = [
        # Low-risk cases
        {"name": "Case 1: Young Healthy Male", "expected_risk": 10.0, "data": {"age": 30, "sex": 1, "cp": 0, "trestbps": 110, "chol": 180, "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 2: Young Healthy Female", "expected_risk": 8.0, "data": {"age": 28, "sex": 0, "cp": 0, "trestbps": 100, "chol": 160, "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 3: Middle-aged Normal", "expected_risk": 15.0, "data": {"age": 40, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 4: Fit Older Female", "expected_risk": 12.0, "data": {"age": 50, "sex": 0, "cp": 0, "trestbps": 115, "chol": 190, "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 5: Young Athlete", "expected_risk": 5.0, "data": {"age": 25, "sex": 1, "cp": 0, "trestbps": 105, "chol": 170, "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},

        # Moderate-risk cases
        {"name": "Case 6: Middle-aged Mild Symptoms", "expected_risk": 25.0, "data": {"age": 45, "sex": 1, "cp": 1, "trestbps": 130, "chol": 220, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2}},
        {"name": "Case 7: Older with Hypertension", "expected_risk": 30.0, "data": {"age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 240, "fbs": 0, "restecg": 0, "thalach": 145, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}},
        {"name": "Case 8: Female with High Cholesterol", "expected_risk": 28.0, "data": {"age": 50, "sex": 0, "cp": 2, "trestbps": 135, "chol": 260, "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0, "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 2}},
        {"name": "Case 9: Middle-aged with Angina", "expected_risk": 35.0, "data": {"age": 48, "sex": 1, "cp": 2, "trestbps": 125, "chol": 230, "fbs": 0, "restecg": 1, "thalach": 135, "exang": 1, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 2}},
        {"name": "Case 10: Older with Mild ECG Issue", "expected_risk": 32.0, "data": {"age": 60, "sex": 1, "cp": 0, "trestbps": 145, "chol": 250, "fbs": 0, "restecg": 2, "thalach": 130, "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2}},

        # High-risk cases
        {"name": "Case 11: Elderly with Angina", "expected_risk": 70.0, "data": {"age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 280, "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 3}},
        {"name": "Case 12: High-risk with Diabetes", "expected_risk": 80.0, "data": {"age": 70, "sex": 1, "cp": 3, "trestbps": 170, "chol": 300, "fbs": 1, "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 3, "thal": 3}},
        {"name": "Case 13: Female with Severe Symptoms", "expected_risk": 65.0, "data": {"age": 62, "sex": 0, "cp": 3, "trestbps": 155, "chol": 290, "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1, "oldpeak": 2.2, "slope": 2, "ca": 2, "thal": 3}},
        {"name": "Case 14: Elderly with Blocked Arteries", "expected_risk": 85.0, "data": {"age": 68, "sex": 1, "cp": 3, "trestbps": 165, "chol": 310, "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 2, "ca": 3, "thal": 3}},
        {"name": "Case 15: High-risk Middle-aged", "expected_risk": 60.0, "data": {"age": 55, "sex": 1, "cp": 3, "trestbps": 150, "chol": 270, "fbs": 1, "restecg": 1, "thalach": 130, "exang": 1, "oldpeak": 1.8, "slope": 2, "ca": 2, "thal": 3}},

        # Edge cases
        {"name": "Case 16: Very Young with High Cholesterol", "expected_risk": 15.0, "data": {"age": 20, "sex": 1, "cp": 0, "trestbps": 110, "chol": 300, "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 17: Elderly Healthy", "expected_risk": 20.0, "data": {"age": 75, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0, "oldpeak": 0, "slope": 0, "ca": 0, "thal": 0}},
        {"name": "Case 18: Young with Angina", "expected_risk": 30.0, "data": {"age": 35, "sex": 1, "cp": 3, "trestbps": 130, "chol": 220, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 1, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}},
        {"name": "Case 19: Female with Diabetes", "expected_risk": 40.0, "data": {"age": 50, "sex": 0, "cp": 0, "trestbps": 140, "chol": 250, "fbs": 1, "restecg": 1, "thalach": 135, "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2}},
        {"name": "Case 20: Max Age High Risk", "expected_risk": 90.0, "data": {"age": 80, "sex": 1, "cp": 3, "trestbps": 180, "chol": 320, "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1, "oldpeak": 3.5, "slope": 2, "ca": 4, "thal": 3}},
        {"name": "Case 21: Low BP High Risk", "expected_risk": 70.0, "data": {"age": 60, "sex": 1, "cp": 3, "trestbps": 100, "chol": 280, "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 3}},
        {"name": "Case 22: High Fitness High Risk", "expected_risk": 60.0, "data": {"age": 55, "sex": 1, "cp": 3, "trestbps": 150, "chol": 270, "fbs": 1, "restecg": 2, "thalach": 170, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 3}},
        {"name": "Case 23: Young with ECG Issue", "expected_risk": 25.0, "data": {"age": 30, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 2, "thalach": 160, "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2}},
        {"name": "Case 24: Middle-aged Extreme Cholesterol", "expected_risk": 45.0, "data": {"age": 45, "sex": 1, "cp": 0, "trestbps": 130, "chol": 400, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 1, "thal": 2}},
        {"name": "Case 25: Elderly Moderate Symptoms", "expected_risk": 50.0, "data": {"age": 70, "sex": 1, "cp": 2, "trestbps": 150, "chol": 260, "fbs": 0, "restecg": 1, "thalach": 130, "exang": 1, "oldpeak": 1.8, "slope": 1, "ca": 1, "thal": 2}}
    ]

    # Run tests
    for case in test_cases:
        test_prediction(case["data"], case["name"], case["expected_risk"])

    print("Completed automated prediction tests")
    logger.info("Completed automated prediction tests")

if __name__ == '__main__':
    main()