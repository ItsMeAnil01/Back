import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import logging
import os
from typing import Tuple

# Set up logging to track data processing and training
logging.basicConfig(
    filename='train_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define expected features to match app.py
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean dataset.
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


def train_model(data: pd.DataFrame) -> Tuple[XGBClassifier, StandardScaler]:
    """
    Train XGBoost model with SMOTE and hyperparameter tuning.
    Returns trained model and scaler.
    """
    try:
        # Select features and target
        X = data[EXPECTED_FEATURES]
        y = data['target']
        logger.info(f"Using features: {EXPECTED_FEATURES}")

        # Handle class imbalance with SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=72)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Applied SMOTE: {len(X_resampled)} samples after resampling")

        # Split dataset (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=72
        )
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Apply scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied StandardScaler to features")

        # Define hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_lambda': [1, 2]
        }

        # Initialize XGBoost model
        xgb = XGBClassifier(random_state=72, eval_metric='logloss')
        grid_search = GridSearchCV(
            xgb, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")

        # Evaluate model on train and test sets
        y_train_pred = best_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred)

        logger.info(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        logger.info(f"Classification Report:\n{report}")

        return best_model, scaler

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
        data = load_and_clean_data('HeartDiseaseRiskAssessment.csv')

        # Train model
        model, scaler = train_model(data)

        # Add version to model
        model.version = "1.0.0"

        # Save model and scaler
        joblib.dump(model, 'heart_attack_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Saved model and scaler to heart_attack_model.pkl and scaler.pkl")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()