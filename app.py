from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import tensorflow as tf
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load models
ann_model = tf.keras.models.load_model('ANN_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError})
scaler = joblib.load('scaler.pkl')

def predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi):
    try:
        input_data = pd.DataFrame({
            'Month': [month],
            'Day': [day],
            'Hour': [hour],
            'Temperature': [temperature],
            'Relative Humidity': [humidity],
            'GHI': [ghi]
        })

        # Ensure compatibility with scaler
        input_scaled = scaler.transform(input_data)

        # Predict tilt angle using ANN model
        predicted_tilt_angle = model.predict(input_scaled)[0][0]

        # Adjust angle for morning hours
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle

        return float(predicted_tilt_angle)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract query parameters
        month = request.args.get('month', type=int)
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        temperature = request.args.get('temperature', type=float)
        humidity = request.args.get('humidity', type=float)
        ghi = request.args.get('ghi', type=float)

        # Ensure all params are present
        if None in (month, day, hour, temperature, humidity, ghi):
            return jsonify({'error': 'Missing or invalid query parameters'}), 400

        tilt_angle = predict_tilt_angle(ann_model, month, day, hour, temperature, humidity, ghi)

        if tilt_angle is None:
            return jsonify({'error': 'Error in prediction'}), 500

        return jsonify({'angle': tilt_angle})

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500
