import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/static_gesture_model.keras"))
CLASSES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/static_class_names.pkl"))

def normalize_keypoints(keypoints):
    """Normalize keypoints relative to the wrist to make the model position-independent."""
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

def evaluate():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("Model or class names file not found. Please train the model first.")
        return

    print("Loading model and classes...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = joblib.load(CLASSES_PATH)

    print("Loading dataset for evaluation...")
    X, y_true = [], []
    for label, gesture in enumerate(class_names):
        gesture_dir = os.path.join(DATASET_FOLDER, gesture)
        if not os.path.exists(gesture_dir):
            continue
        for filename in os.listdir(gesture_dir):
            if filename.endswith(".npy"):
                file_path = os.path.join(gesture_dir, filename)
                try:
                    keypoints = np.load(file_path)
                    X.append(normalize_keypoints(keypoints))
                    y_true.append(label)
                except Exception:
                    pass

    if len(X) == 0:
        print("No validation data found.")
        return

    X = np.array(X)
    y_true = np.array(y_true)

    print(f"Running predictions on {len(X)} samples...")
    # Predict all samples
    predictions = model.predict(X, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
    print("="*50 + "\n")
    
    print("Detailed Classification Report (Per-Sign Accuracy):")
    try:
        present_labels = sorted(list(set(y_true)))
        present_names = [class_names[i] for i in present_labels]
        report = classification_report(y_true, y_pred, labels=present_labels, target_names=present_names)
        print(report)
    except Exception as e:
        print(f"Could not generate detailed report: {e}")

if __name__ == "__main__":
    evaluate()
