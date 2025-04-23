import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import logging
import os
from typing import Tuple, Any, List

# Set up logging
logging.basicConfig(
    filename='train_model.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}")
    logger.debug(f"Attempting to load data from {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
        print(f"Loaded dataset with {len(data)} rows")
        missing_features = [f for f in EXPECTED_FEATURES if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features in dataset: {missing_features}")
            raise ValueError(f"Missing features: {missing_features}")
        if 'target' not in data.columns:
            logger.error("Target column 'target' not found in dataset")
            raise ValueError("Target column 'target' not found in dataset")
        data['oldpeak'] = data['oldpeak'].round(1)
        logger.info("Rounded oldpeak to 1 decimal place")
        missing_values = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_values}")
        data[EXPECTED_FEATURES] = data[EXPECTED_FEATURES].fillna(data[EXPECTED_FEATURES].mean())
        missing_values_after = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values after imputation:\n{missing_values_after}")
        logger.info(f"Dataset summary:\n{data[EXPECTED_FEATURES].describe()}")
        logger.info(f"Target distribution:\n{data['target'].value_counts()}")
        return data
    except Exception as e:
        logger.error(f"Failed to load and clean data: {str(e)}")
        print(f"Error loading data: {str(e)}")
        raise

def train_model(data: pd.DataFrame) -> Tuple[Any, StandardScaler, List[str]]:
    print("Starting model training")
    logger.debug("Starting model training")
    try:
        X = data[EXPECTED_FEATURES]
        y = data['target']
        logger.info(f"Using features: {EXPECTED_FEATURES}")

        # Feature selection using Random Forest
        rf = RandomForestClassifier(random_state=72)
        rf.fit(X, y)
        feature_importance = pd.Series(rf.feature_importances_, index=EXPECTED_FEATURES)
        logger.info(f"Feature Importance:\n{feature_importance.sort_values(ascending=False)}")
        selected_features = EXPECTED_FEATURES  # Use all for now, adjust if needed
        X = X[selected_features]

        # Apply SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=72)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Applied SMOTE: {len(X_resampled)} samples after resampling")
        print(f"SMOTE applied: {len(X_resampled)} samples")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=72
        )
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Apply scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied StandardScaler to features")

        # Define models and parameter grids
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=72),
                'param_grid': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=72, eval_metric='logloss'),
                'param_grid': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=72, max_iter=2000),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear', 'saga'],
                    'penalty': ['l2']
                }
            }
        }

        best_model = None
        best_accuracy = 0
        best_model_name = ''

        # Train and evaluate each model
        for name, config in models.items():
            print(f"Training {name}...")
            logger.debug(f"Starting GridSearchCV for {name}")
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best hyperparameters for {name}: {grid_search.best_params_}")
            print(f"Best hyperparameters for {name}: {grid_search.best_params_}")

            # Evaluate on test set
            y_test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred)
            logger.info(f"{name} Test Accuracy: {test_accuracy * 100:.2f}%")
            logger.info(f"{name} Classification Report:\n{report}")
            print(f"{name} Test Accuracy: {test_accuracy * 100:.2f}%")

            # Update best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = model
                best_model_name = name

        logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy * 100:.2f}%")
        print(f"Best model: {best_model_name} with accuracy {best_accuracy * 100:.2f}%")
        return best_model, scaler, selected_features
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        print(f"Error training model: {str(e)}")
        raise

def main():
    print("Starting train_model.py")
    logger.debug("Starting main function")
    try:
        if ' ' in os.getcwd():
            logger.warning(f"Project directory contains spaces: {os.getcwd()}")
            print(f"Warning: Project directory contains spaces: {os.getcwd()}")
        data = load_and_clean_data('heart.csv')
        model, scaler, selected_features = train_model(data)
        model.version = "1.0.0"
        joblib.dump(model, 'heart_attack_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(selected_features, 'selected_features.pkl')
        logger.info("Saved model, scaler, and selected features to heart_attack_model.pkl, scaler.pkl, and selected_features.pkl")
        print("Saved model, scaler, and selected features to heart_attack_model.pkl, scaler.pkl, and selected_features.pkl")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()