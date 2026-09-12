from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model weights
MODEL_PATH = "models/hybrid_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ["Class_0", "Class_1", "Class_2"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "Hybrid CNN-RNN-ANN Inference API",
        "platform": "Kaggle Cloud GPU"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' key in payload"}), 400

        seq = np.array(data["sequence"], dtype=np.float32)
        if seq.ndim == 4:
            seq = np.expand_dims(seq, axis=0)

        preds = model.predict(seq, verbose=0)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds, axis=1)[0])

        return jsonify({
            "predicted_index": pred_idx,
            "label": CLASSES[pred_idx] if pred_idx < len(CLASSES) else str(pred_idx),
            "confidence": round(confidence, 4)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
