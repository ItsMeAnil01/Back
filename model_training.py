import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
from typing import Tuple

# Set up logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected features to match app.py
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean heart.csv dataset.
    Returns cleaned DataFrame.
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load dataset
        data = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")

        # Check for expected features and target
        missing_features = [f for f in EXPECTED_FEATURES if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features in dataset: {missing_features}")
            raise ValueError(f"Dataset missing features: {missing_features}")
        if 'target' not in data.columns:
            logger.error("Target column 'target' not found in dataset")
            raise ValueError("Target column 'target' not found in dataset")

        # Clean oldpeak: round to 1 decimal place
        data['oldpeak'] = data['oldpeak'].round(1)
        logger.info("Rounded oldpeak to 1 decimal place")

        # Handle missing values: impute with mean
        missing_values = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_values}")
        data[EXPECTED_FEATURES] = data[EXPECTED_FEATURES].fillna(data[EXPECTED_FEATURES].mean())
        missing_values_after = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values after imputation:\n{missing_values_after}")

        # Log dataset summary and target distribution
        logger.info(f"Dataset summary:\n{data[EXPECTED_FEATURES].describe()}")
        logger.info(f"Target distribution:\n{data['target'].value_counts()}")

        return data

    except Exception as e:
        logger.error(f"Failed to load and clean data: {str(e)}")
        raise


def train_model(data: pd.DataFrame) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression model.
    Returns trained model and scaler.
    """
    try:
        # Select features and target
        X = data[EXPECTED_FEATURES]
        y = data['target']
        logger.info(f"Using features: {EXPECTED_FEATURES}")

        # Split dataset (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=72
        )
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Apply scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied StandardScaler to features")

        # Train logistic regression model
        model = LogisticRegression(random_state=72, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        logger.info("Trained logistic regression model")

        # Evaluate model
        y_train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred)

        logger.info(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        logger.info(f"Classification Report:\n{report}")

        return model, scaler

    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise


def main():
    """Main function to load data, train model, and save artifacts."""
    try:
        # Check project directory for spaces
        if ' ' in os.getcwd():
            logger.warning(f"Project directory contains spaces: {os.getcwd()}. Consider renaming to avoid issues.")

        # Load and clean data
        data = load_and_clean_data('heart.csv')

        # Train model
        model, scaler = train_model(data)

        # Save model and scaler
        joblib.dump(model, 'heart_attack_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Saved model and scaler to heart_attack_model.pkl and scaler.pkl")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()