import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Sequence & Frame dimensions
SEQ_LEN = 10         # Frames per video sequence
IMG_HEIGHT = 64      # Resized frame height
IMG_WIDTH = 64       # Resized frame width
CHANNELS = 3         # RGB channels

def extract_frames(video_path, seq_len=SEQ_LEN):
    """Extracts a fixed number of uniformly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < seq_len:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    frames = []
    current_idx = 0
    
    while cap.isOpened() and len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
        current_idx += 1
        
    cap.release()
    return np.array(frames) if len(frames) == seq_len else None

def build_hybrid_model(seq_len=SEQ_LEN, height=IMG_HEIGHT, width=IMG_WIDTH, channels=CHANNELS, num_classes=3):
    """
    Hybrid Architecture:
    1. CNN (TimeDistributed): Extracts spatial features from individual frames.
    2. RNN (LSTM): Captures temporal movement patterns across consecutive frames.
    3. ANN (Dense): Performs final multi-class classification.
    """
    inputs = layers.Input(shape=(seq_len, height, width, channels))

    # --- 1. CNN Feature Extractor (Spatial) ---
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # --- 2. RNN Layer (Temporal sequence processing) ---
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # --- 3. ANN Dense Classification Block ---
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_RNN_ANN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    print("Building Hybrid CNN-RNN-ANN Architecture...")
    model = build_hybrid_model()
    model.summary()
    
    save_path = "models/hybrid_model.keras"
    model.save(save_path)
    print(f"\nModel compiled and saved successfully to: {save_path}")