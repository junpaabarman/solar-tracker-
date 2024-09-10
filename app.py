from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd
import os

# Custom loss function for the ANN model
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the ANN model and the scaler
model = tf.keras.models.load_model('ANN_model.h5', custom_objects={'mse': mse})
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

def predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi):
    try:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Month': [month],
            'Day': [day],
            'Hour': [hour],
            'Temperature': [temperature],
            'Relative Humidity': [humidity],
            'GHI': [ghi]
        })

        # Scale the input data using the pre-fitted scaler
        input_scaled = scaler.transform(input_data)

        # Predict the tilt angle using the ANN model
        predicted_tilt_angle = model.predict(input_scaled)[0][0]

        # Adjust the predicted angle based on the time of day (for tilt optimization)
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle

        return float(predicted_tilt_angle)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Serve the static home page (index.html)
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

# Prediction route
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve query parameters
        month = request.args.get('month', type=int)
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        temperature = request.args.get('temperature', type=float)
        humidity = request.args.get('humidity', type=float)
        ghi = request.args.get('ghi', type=float)

        # Check for missing parameters
        if None in (month, day, hour, temperature, humidity, ghi):
            return jsonify({'error': 'Missing or invalid query parameters'}), 400

        # Predict the tilt angle
        tilt_angle = predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi)

        if tilt_angle is None:
            return jsonify({'error': 'Error in prediction'}), 500

        return jsonify({'angle': tilt_angle})

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
#if __name__ == '__main__':
   # app.run(debug=True)
