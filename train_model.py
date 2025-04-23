import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import logging
import os
from typing import Tuple, Any

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

NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


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

        # Round oldpeak to 1 decimal place
        data['oldpeak'] = data['oldpeak'].round(1)
        logger.info("Rounded oldpeak to 1 decimal place")

        # Impute missing values: mode for categorical, median for numerical
        missing_values = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_values}")
        for feature in CATEGORICAL_FEATURES:
            data[feature] = data[feature].fillna(data[feature].mode()[0])
        for feature in NUMERICAL_FEATURES:
            data[feature] = data[feature].fillna(data[feature].median())
        missing_values_after = data[EXPECTED_FEATURES].isnull().sum()
        logger.info(f"Missing values after imputation:\n{missing_values_after}")

        logger.info(f"Dataset summary:\n{data[EXPECTED_FEATURES].describe()}")
        logger.info(f"Target distribution:\n{data['target'].value_counts()}")
        return data
    except Exception as e:
        logger.error(f"Failed to load and clean data: {str(e)}")
        print(f"Error loading data: {str(e)}")
        raise


def train_model(data: pd.DataFrame) -> Tuple[Any, ColumnTransformer]:
    print("Starting model training")
    logger.debug("Starting model training")
    try:
        X = data[EXPECTED_FEATURES]
        y = data['target']
        logger.info(f"Using features: {EXPECTED_FEATURES}")

        # Split dataset with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=72, stratify=y
        )
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Define preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
            ]
        )

        # Define models and parameter grids
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=72),
                'param_grid': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=72, eval_metric='logloss'),
                'param_grid': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__subsample': [0.7, 0.8, 0.9],
                    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=72, max_iter=2000),
                'param_grid': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__solver': ['lbfgs', 'liblinear'],
                    'classifier__penalty': ['l2']
                }
            }
        }

        best_model = None
        best_accuracy = 0
        best_roc_auc = 0
        best_model_name = ''
        best_calibration = ''

        # Train and evaluate each model with different calibration methods
        for name, config in models.items():
            for calibration_method in ['sigmoid', 'isotonic']:
                print(f"Training {name} with {calibration_method} calibration...")
                logger.debug(f"Starting GridSearchCV for {name} with {calibration_method}")

                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', config['model'])
                ])

                # Perform GridSearchCV
                grid_search = GridSearchCV(
                    pipeline,
                    config['param_grid'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best hyperparameters for {name}: {grid_search.best_params_}")
                print(f"Best hyperparameters for {name}: {grid_search.best_params_}")

                # Calibrate probabilities
                print(f"Calibrating {name} with {calibration_method}...")
                calibrated_model = CalibratedClassifierCV(
                    model, cv=5, method=calibration_method
                )
                calibrated_model.fit(X_train, y_train)
                logger.info(f"Calibrated {name} with {calibration_method}")

                # Evaluate on test set
                y_test_pred = calibrated_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                roc_auc = roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1])
                report = classification_report(y_test, y_test_pred)
                logger.info(f"{name} ({calibration_method}) Test Accuracy: {test_accuracy * 100:.2f}%")
                logger.info(f"{name} ({calibration_method}) ROC-AUC: {roc_auc:.2f}")
                logger.info(f"{name} ({calibration_method}) Classification Report:\n{report}")
                print(f"{name} ({calibration_method}) Test Accuracy: {test_accuracy * 100:.2f}%")
                print(f"{name} ({calibration_method}) ROC-AUC: {roc_auc:.2f}")

                # Update best model (prioritize accuracy, then ROC-AUC)
                if test_accuracy > best_accuracy or (
                        test_accuracy == best_accuracy and roc_auc > best_roc_auc
                ):
                    best_accuracy = test_accuracy
                    best_roc_auc = roc_auc
                    best_model = calibrated_model
                    best_model_name = name
                    best_calibration = calibration_method

        logger.info(
            f"Best model: {best_model_name} with {best_calibration} calibration, accuracy {best_accuracy * 100:.2f}%, ROC-AUC {best_roc_auc:.2f}")
        print(
            f"Best model: {best_model_name} with {best_calibration} calibration, accuracy {best_accuracy * 100:.2f}%, ROC-AUC {best_roc_auc:.2f}")
        return best_model, preprocessor
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
        model, preprocessor = train_model(data)
        model.version = "1.0.0"
        joblib.dump(model, 'heart_attack_model.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        logger.info("Saved model and preprocessor to heart_attack_model.pkl and preprocessor.pkl")
        print("Saved model and preprocessor to heart_attack_model.pkl and preprocessor.pkl")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()