from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['GET'])
def predict():
    # Simulate a prediction without actually loading or using the model
    return jsonify({'angle': 45.0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
