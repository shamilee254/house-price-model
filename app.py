from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# Load model file from the same folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "housing_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        size = float(data["size"])
    except Exception:
        return jsonify({"error": "Invalid or missing 'size'"}), 400

    try:
        pred = model.predict([[size]])
        price = float(np.asarray(pred).ravel()[0])
        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
