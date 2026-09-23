import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))
MODEL_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../../models"))

def normalize_keypoints(keypoints):
    """Normalize keypoints relative to the wrist to make the model position-independent."""
    normalized = np.zeros_like(keypoints)
    
    # Process Hand 1
    if np.any(keypoints[0:63]): # If hand exists
        wrist_x, wrist_y, wrist_z = keypoints[0], keypoints[1], keypoints[2]
        for i in range(21):
            normalized[i*3] = keypoints[i*3] - wrist_x
            normalized[i*3+1] = keypoints[i*3+1] - wrist_y
            normalized[i*3+2] = keypoints[i*3+2] - wrist_z
            
    # Process Hand 2
    if np.any(keypoints[63:126]): # If hand exists
        wrist_x, wrist_y, wrist_z = keypoints[63], keypoints[64], keypoints[65]
        for i in range(21):
            offset = 63 + (i*3)
            normalized[offset] = keypoints[offset] - wrist_x
            normalized[offset+1] = keypoints[offset+1] - wrist_y
            normalized[offset+2] = keypoints[offset+2] - wrist_z
            
    return normalized

def train_model():
    if not os.path.exists(DATASET_FOLDER):
        print(f"Error: Dataset folder not found at {DATASET_FOLDER}")
        return

    classes = sorted([d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
    if len(classes) < 2:
        print("Error: Need at least 2 gesture classes to train.")
        return

    print(f"Found {len(classes)} classes: {classes}")

    X, y = [], []
    for label, gesture in enumerate(classes):
        gesture_dir = os.path.join(DATASET_FOLDER, gesture)
        for filename in os.listdir(gesture_dir):
            if filename.endswith(".npy"):
                file_path = os.path.join(gesture_dir, filename)
                try:
                    keypoints = np.load(file_path)
                    X.append(keypoints)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    # Apply normalization to all sequences
    X_normalized = np.array([normalize_keypoints(kp) for kp in X])
    
    if len(X_normalized) == 0:
        print("No valid data found to train.")
        return

    print(f"Loaded {len(X_normalized)} static frames across {len(classes)} classes.")
    
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(classes, os.path.join(MODEL_FOLDER, "static_class_names.pkl"))
    
    y = to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # Static DNN Model Architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(126,)), # 21 * 3 * 2 (Two Hands)
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training Static Keypoint Model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    model_path = os.path.join(MODEL_FOLDER, "static_gesture_model.keras")
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()
