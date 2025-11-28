from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
import logging # Import logging for better error handling/debugging

# Set up logging for Render logs
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# Load model file from the same folder
# Ensure 'housing_model.pkl' is also committed and pushed to your repo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "housing_model.pkl")

# Use a try-except block to handle the case where the model file might be missing during deployment
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
    MODEL_LOADED = True
except FileNotFoundError:
    logging.error(f"ERROR: Model file not found at {MODEL_PATH}")
    MODEL_LOADED = False
    model = None
except Exception as e:
    logging.error(f"ERROR: Failed to load model: {e}")
    MODEL_LOADED = False
    model = None


# --- NEW: Health Check/Root Route ---
@app.route("/", methods=["GET"])
def home():
    """Returns a simple status message for the root URL."""
    if MODEL_LOADED:
        return jsonify({"status": "ok", "message": "House Price Prediction API is running. Use /predict for predictions."})
    else:
        return jsonify({"status": "error", "message": "API is running but failed to load the prediction model."}), 500
# -----------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    """Handles POST requests to make price predictions."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500
        
    data = request.get_json(force=True)
    
    # 1. Input Validation
    try:
        size = float(data.get("size"))
    except TypeError:
        # Handle case where 'size' is missing or not a valid number
        logging.error("Invalid or missing 'size' in request.")
        return jsonify({"error": "Invalid or missing 'size'. Please provide a numeric value."}), 400

    # 2. Prediction Logic
    try:
        # The model expects a 2D array, so we wrap the single input
        input_data = np.array([[size]]) 
        
        pred = model.predict(input_data)
        
        # Extract the prediction result (assuming it returns a single value)
        price = float(np.asarray(pred).ravel()[0])
        
        logging.info(f"Prediction made for size {size}: {price}")
        return jsonify({"price": price})
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed due to an internal server error."}), 500

# The Render environment automatically sets the PORT, so the `if __name__ == "__main__":` block 
# is typically ignored when using Gunicorn, but it's good practice to keep for local testing.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)