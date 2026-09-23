import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

BASE_DIR = r"d:\Gesture Voice AI\EDUVISIONassistant\ai\gesture_training"
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/static_gesture_model.keras"))

def normalize_keypoints(keypoints):
    normalized = np.zeros_like(keypoints)
    if np.any(keypoints[0:63]): 
        wrist_x, wrist_y, wrist_z = keypoints[0], keypoints[1], keypoints[2]
        for i in range(21):
            normalized[i*3] = keypoints[i*3] - wrist_x
            normalized[i*3+1] = keypoints[i*3+1] - wrist_y
            normalized[i*3+2] = keypoints[i*3+2] - wrist_z
    if np.any(keypoints[63:126]): 
        wrist_x, wrist_y, wrist_z = keypoints[63], keypoints[64], keypoints[65]
        for i in range(21):
            offset = 63 + (i*3)
            normalized[offset] = keypoints[offset] - wrist_x
            normalized[offset+1] = keypoints[offset+1] - wrist_y
            normalized[offset+2] = keypoints[offset+2] - wrist_z
    return normalized

classes = sorted([d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
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
                pass

X = np.array(X)
y = np.array(y)
X_normalized = np.array([normalize_keypoints(kp) for kp in X])
y = to_categorical(y).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

model = load_model(MODEL_PATH)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Static Model Accuracy: {acc*100:.2f}%")
print(f"Static Model Loss: {loss:.4f}")
