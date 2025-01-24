from flask import Flask, request, jsonify
from app.model import DowntimeModel

app = Flask(__name__)
model = DowntimeModel()

@app.route('/train', methods=['POST'])
def train_model():
    # Train the model on the provided dataset
    dataset_path = request.json.get('dataset_path', 'data/dataset.csv')
    report = model.train(dataset_path)
    return jsonify({"message": "Model trained successfully", "report": report})

@app.route('/predict', methods=['POST'])
def predict():
    # Predict based on input data
    input_data = request.json.get('features', [])
    result = model.predict(input_data)
    return jsonify(result)
