from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset
csv_file = "heart.csv"
df = pd.read_csv(csv_file)

# Ensure dataset has all required features
expected_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal", "extra1", "extra2"]
for feature in expected_features:
    if feature not in df.columns:
        df[feature] = 0  # Fill missing columns with 0

# Separate features and target
X = df[expected_features]
y = df['target'] if 'target' in df.columns else np.random.randint(0, 2, size=len(df))

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")


def train_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_scaled, y)
    joblib.dump(model, "heart_attack_model.pkl")


train_model()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/submit', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received Input Data:", data)  # Debugging

        # Convert input data to DataFrame and ensure all values are float
        input_data = pd.DataFrame([{key: float(value) for key, value in data.items()}])

        # Ensure all expected features exist
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Load model & scaler
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("heart_attack_model.pkl")

        # Scale and predict
        input_scaled = scaler.transform(input_data[expected_features])
        prediction = model.predict_proba(input_scaled)[0, 1] * 100

        return jsonify({"prediction": f"Heart Attack Risk: {prediction:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/train', methods=['GET'])
def retrain():
    train_model()
    return "Model retrained successfully!"


if __name__ == '__main__':
    app.run(debug=True)
