import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Hyperparameters ---
SEQ_LEN = 20       # 20 temporal frames per video
IMG_SIZE = 96      # 96x96 spatial resolution
CLASSES = ["CricketShot", "Punch", "TennisSwing"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASSES)}

CSV_PATH = os.path.join("data", "train.csv")
TRAIN_DIR = os.path.join("data", "train")
MODEL_SAVE_PATH = os.path.join("models", "hybrid_model.keras")

def extract_dense_frames(video_path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    """Uniformly extracts seq_len frames at (img_size, img_size)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None

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

def load_data_from_csv():
    """Reads train.csv, filters for target classes, and extracts video sequences."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["tag"].isin(CLASSES)].reset_index(drop=True)
    print(f"Found {len(df)} matching training records for target classes: {CLASSES}")

    X, y = [], []
    for idx, row in df.iterrows():
        video_filename = row["video_name"]
        tag = row["tag"]
        video_path = os.path.join(TRAIN_DIR, video_filename)

        if not os.path.exists(video_path):
            continue

        frames = extract_dense_frames(video_path)
        if frames is not None and len(frames) == SEQ_LEN:
            X.append(frames)
            y.append(LABEL_MAP[tag])

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"Processed {idx + 1}/{len(df)} videos...")

    X = np.array(X, dtype=np.float32)
    y = to_categorical(np.array(y), num_classes=len(CLASSES))
    return X, y

def build_accurate_hybrid_model(seq_len=SEQ_LEN, img_size=IMG_SIZE, num_classes=3):
    model = Sequential([
        # --- Stage 1: Spatial Feature Extraction (CNN) ---
        TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"), input_shape=(seq_len, img_size, img_size, 3)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),

        TimeDistributed(Flatten()),

        # --- Stage 2: Temporal Sequence Learning (RNN / LSTM) ---
        LSTM(64, return_sequences=False),
        Dropout(0.4),

        # --- Stage 3: Classification Network (ANN) ---
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    print("Extracting frames from CSV catalog...")
    X, y = load_data_from_csv()

    if len(X) == 0:
        print("Error: Could not locate any matching video files inside data/train/")
        exit()

    print(f"\nDataset shape: {X.shape}, Labels shape: {y.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    model = build_accurate_hybrid_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    ]

    print("\nInitiating model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=4,
        callbacks=callbacks
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel successfully saved to {MODEL_SAVE_PATH}")
