import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "models/hybrid_model.keras"

# Check if model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python model_builder.py' first.")

print("Loading Hybrid CNN-RNN-ANN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Target labels (update based on your dataset classes)
CLASSES = ["CricketShot", "Punch", "TennisSwing"]

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Hybrid CNN-RNN-ANN Inference API",
        "input_shape": [10, 64, 64, 3]
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in request body"}), 400

        # Convert input payload to numpy array
        seq = np.array(data["sequence"], dtype=np.float32)

        # Ensure shape matches (batch_size, seq_len, height, width, channels)
        if seq.ndim == 4:
            seq = np.expand_dims(seq, axis=0)
        elif seq.ndim != 5:
            return jsonify({
                "error": f"Invalid sequence dimensions. Expected 4D or 5D array, received {seq.ndim}D"
            }), 400

        predictions = model.predict(seq, verbose=0)
        class_idx = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions, axis=1)[0])

        return jsonify({
            "predicted_index": class_idx,
            "label": CLASSES[class_idx] if class_idx < len(CLASSES) else f"Class_{class_idx}",
            "confidence": round(confidence, 4)
        }), 200

    except Exception as err:
        return jsonify({"error": str(err)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
