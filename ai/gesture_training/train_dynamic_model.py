import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data/dynamic_asl')
MODEL_PATH = os.path.join(BASE_DIR, '../../models/dynamic_lstm_model.keras')
ACTIONS = np.array(['hello', 'yes', 'no', 'thanks'])
NO_SEQUENCES = 30     # 30 videos per action
SEQUENCE_LENGTH = 30  # 30 frames per video

def load_data():
    label_map = {label:num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
            
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y

def main():
    print("Loading data from disk...")
    try:
        X, y = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure you have run collect_dynamic_data.py completely first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    print("Building LSTM Model...")
    model = Sequential()
    # 30 frames, 126 features (2 hands * 21 points * 3 coordinates)
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(ACTIONS.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print("Training Model (this should only take a few seconds)...")
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    print(f"Saving Model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    
    print("LSTM Training Complete! You can now run predict_dynamic_lstm.py")

if __name__ == '__main__':
    main()
