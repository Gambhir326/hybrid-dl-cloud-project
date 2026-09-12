import os
import cv2
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "models/hybrid_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python model_builder.py' first.")

print("Loading upgraded Hybrid CNN-RNN-ANN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

CLASSES = ["CricketShot", "Punch", "TennisSwing"]

# Match updated architecture
SEQ_LEN = 20
IMG_SIZE = 96

def extract_frames_from_video(video_path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    """Uniformly extracts seq_len frames at (img_size, img_size)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError("Cannot read video frames or video file is corrupted.")

    step = max(1, total_frames // seq_len)
    frames = []

    for i in range(seq_len):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame / 255.0)

    cap.release()
    return np.array(frames, dtype=np.float32)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(temp_path)

    try:
        sequence = extract_frames_from_video(temp_path, seq_len=SEQ_LEN, img_size=IMG_SIZE)
        input_tensor = np.expand_dims(sequence, axis=0)  # Shape: (1, 20, 96, 96, 3)

        predictions = model.predict(input_tensor, verbose=0)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])

        # Generate complete probability breakdown across all classes
        breakdown = {
            CLASSES[i]: round(float(predictions[i]), 4)
            for i in range(len(CLASSES))
        }

        return jsonify({
            "predicted_index": class_idx,
            "label": CLASSES[class_idx],
            "confidence": round(confidence, 4),
            "breakdown": breakdown
        }), 200

    except Exception as err:
        return jsonify({"error": str(err)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)import os
import cv2
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "models/hybrid_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python model_builder.py' first.")

print("Loading upgraded Hybrid CNN-RNN-ANN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

CLASSES = ["CricketShot", "Punch", "TennisSwing"]

# Match updated architecture
SEQ_LEN = 20
IMG_SIZE = 96

def extract_frames_from_video(video_path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    """Uniformly extracts seq_len frames at (img_size, img_size)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError("Cannot read video frames or video file is corrupted.")

    step = max(1, total_frames // seq_len)
    frames = []

    for i in range(seq_len):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame / 255.0)

    cap.release()
    return np.array(frames, dtype=np.float32)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(temp_path)

    try:
        sequence = extract_frames_from_video(temp_path, seq_len=SEQ_LEN, img_size=IMG_SIZE)
        input_tensor = np.expand_dims(sequence, axis=0)  # Shape: (1, 20, 96, 96, 3)

        predictions = model.predict(input_tensor, verbose=0)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])

        # Generate complete probability breakdown across all classes
        breakdown = {
            CLASSES[i]: round(float(predictions[i]), 4)
            for i in range(len(CLASSES))
        }

        return jsonify({
            "predicted_index": class_idx,
            "label": CLASSES[class_idx],
            "confidence": round(confidence, 4),
            "breakdown": breakdown
        }), 200

    except Exception as err:
        return jsonify({"error": str(err)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
