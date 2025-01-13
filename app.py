from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("model", "iris_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Welcome to the Iris Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get input JSON
        features = np.array(data['features']).reshape(1, -1)  # Reshape input for model
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})  # Return prediction
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
