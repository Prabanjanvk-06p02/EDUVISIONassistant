import os
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../../models"))
os.makedirs(MODEL_FOLDER, exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/sign-language-mnist/sign_mnist_train.csv"
TEST_URL = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/sign-language-mnist/sign_mnist_test.csv"

def prepare_data(df):
    # First column is the label (0-24, where J=9 and Z=25 are excluded due to motion)
    labels = df['label'].values
    # Remaining 784 columns are pixels (28x28)
    images = df.drop('label', axis=1).values
    
    # Reshape and normalize
    images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels (up to 25 to accommodate the max class value)
    labels = to_categorical(labels, num_classes=25)
    
    return images, labels

def main():
    print("=" * 50)
    print("Automated Professional ASL Model Training Pipeline")
    print("=" * 50)
    
    print("1. Downloading high-precision Sign Language MNIST dataset...")
    try:
        train_df = pd.read_csv(TRAIN_URL)
        test_df = pd.read_csv(TEST_URL)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return
        
    print(f"   Successfully loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    
    print("\n2. Processing and normalizing image data...")
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    print("\n3. Building Convolutional Neural Network (CNN)...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(25, activation='softmax') # 25 classes (0-24)
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print("\n4. Training the model (this will take about 1-2 minutes)...")
    # Using 10 epochs for a good balance of speed and professional accuracy
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("\n5. Evaluating model accuracy on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Professional Test Accuracy Achieved: {test_acc * 100:.2f}%")
    
    model_path = os.path.join(MODEL_FOLDER, "asl_cnn_model.keras")
    model.save(model_path)
    print(f"\n✅ Training Complete! Model automatically saved to: {model_path}")
    print("You can now run 'python ai/gesture_prediction/predict_professional_cnn.py' to use it.")

if __name__ == "__main__":
    main()
