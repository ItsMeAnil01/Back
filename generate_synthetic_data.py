import pandas as pd
import numpy as np
import logging
import os
from scipy.stats import gaussian_kde

# Set up logging
logging.basicConfig(
    filename='generate_synthetic_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature ranges from app.py
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
    "thal": (0, 3),
    "target": (0, 1)
}

# Features and target
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def generate_synthetic_data(input_file: str, output_file: str, num_samples: int = 1000):
    """
    Generate synthetic data based on input CSV and save to output CSV.
    """
    try:
        # Read input CSV
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        data = pd.read_csv(input_file)
        logger.info(f"Loaded input data with {len(data)} rows and {len(data.columns)} columns")

        # Verify columns
        missing_cols = [col for col in COLUMNS if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns in input data: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Compute KDE for continuous features
        kde_estimators = {}
        for col in CONTINUOUS_FEATURES:
            values = data[col].dropna().values
            kde = gaussian_kde(values)
            kde_estimators[col] = kde
            logger.info(f"Computed KDE for {col}")

        # Compute probabilities for categorical features
        cat_probs = {}
        for col in CATEGORICAL_FEATURES:
            value_counts = data[col].value_counts(normalize=True)
            values = value_counts.index
            probs = value_counts.values
            cat_probs[col] = {'values': values, 'probs': probs}
            logger.info(f"Probabilities for {col}: {dict(zip(values, probs))}")

        # Generate synthetic data
        synthetic_data = {}
        for col in COLUMNS:
            if col in CONTINUOUS_FEATURES:
                # Continuous features: sample from KDE
                samples = kde_estimators[col].resample(num_samples)[0]
                samples = np.clip(samples, FEATURE_RANGES[col][0], FEATURE_RANGES[col][1])
                if col in ['age', 'trestbps', 'chol', 'thalach']:
                    samples = np.round(samples).astype(int)
                elif col == 'oldpeak':
                    samples = np.round(samples, 1)
            else:
                # Categorical features: sample from empirical distribution
                samples = np.random.choice(
                    cat_probs[col]['values'],
                    size=num_samples,
                    p=cat_probs[col]['probs']
                ).astype(int)
            synthetic_data[col] = samples
        synthetic_df = pd.DataFrame(synthetic_data, columns=COLUMNS)
        logger.info(f"Generated synthetic data with {len(synthetic_df)} rows")

        # Save to output CSV with permission handling
        output_dir = os.path.dirname(output_file) or '.'
        if not os.access(output_dir, os.W_OK):
            logger.error(f"No write permission in directory: {output_dir}")
            raise PermissionError(f"No write permission in {output_dir}")
        try:
            synthetic_df.to_csv(output_file, index=False)
            logger.info(f"Saved synthetic data to {output_file}")
        except PermissionError:
            logger.error(f"Permission denied: Unable to write to {output_file}")
            raise PermissionError(f"Permission denied: Unable to write to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {str(e)}")
        raise

def main():
    try:
        input_file = 'heart.csv'
        output_file = 'synthetic_heart.csv'
        generate_synthetic_data(input_file, output_file, num_samples=1000)
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()