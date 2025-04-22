import pandas as pd
import numpy as np
import logging
import os

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

        # Compute statistics
        stats = {}
        for col in COLUMNS:
            mean = data[col].mean()
            std = data[col].std()
            min_val, max_val = FEATURE_RANGES[col]
            stats[col] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
            logger.info(f"Statistics for {col}: mean={mean:.2f}, std={std:.2f}, range=({min_val}, {max_val})")

        # Generate synthetic data
        synthetic_data = {}
        for col in COLUMNS:
            # Generate samples from normal distribution
            samples = np.random.normal(stats[col]['mean'], stats[col]['std'], num_samples)
            # Clip to feature range and round appropriately
            samples = np.clip(samples, stats[col]['min'], stats[col]['max'])
            if col in ['age', 'trestbps', 'chol', 'thalach', 'ca']:
                samples = np.round(samples).astype(int)
            elif col == 'oldpeak':
                samples = np.round(samples, 1)
            else:
                samples = np.round(samples).astype(int)
            synthetic_data[col] = samples
        synthetic_df = pd.DataFrame(synthetic_data, columns=COLUMNS)
        logger.info(f"Generated synthetic data with {len(synthetic_df)} rows")

        # Save to output CSV
        synthetic_df.to_csv(output_file, index=False)
        logger.info(f"Saved synthetic data to {output_file}")

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