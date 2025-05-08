import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle

# Generate synthetic dataset for heart disease prediction
np.random.seed(42)
n_samples = 100  # Increased for better training
data = {
    'age': np.random.randint(29, 77, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(94, 200, n_samples),
    'chol': np.random.randint(126, 564, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(71, 202, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6.2, n_samples),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 5, n_samples),
    'thal': np.random.randint(0, 4, n_samples),
    'target': np.random.randint(0, 2, n_samples)  # Binary target (0 = no disease, 1 = disease)
}

# Include the example from logs as the first sample
data['age'][0] = 45
data['sex'][0] = 1
data['cp'][0] = 0
data['trestbps'][0] = 120
data['chol'][0] = 200
data['fbs'][0] = 0
data['restecg'][0] = 0
data['thalach'][0] = 150
data['exang'][0] = 0
data['oldpeak'][0] = 0
data['slope'][0] = 0
data['ca'][0] = 0
data['thal'][0] = 0
data['target'][0] = 0

df = pd.DataFrame(data)

# Define features and target
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

# Define preprocessing pipeline
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ],
    remainder='passthrough'
)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save preprocessor and model
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline.named_steps['model'], f)

print("Model and preprocessor saved as 'model.pkl' and 'preprocessor.pkl'")