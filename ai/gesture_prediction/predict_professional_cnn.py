import cv2
import numpy as np
import tensorflow as tf
import os
import string
import mediapipe as mp
import threading

try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if genai:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/asl_cnn_model.keras"))

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Professional CNN Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run 'python ai/gesture_training/auto_train_professional.py' first!")
    exit(1)

# ASL Alphabet Map
# 0=A, 1=B, ..., 8=I, 9=J (skipped in training), 10=K, ..., 24=Y
alphabet = list(string.ascii_uppercase)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Sentence Builder
sentence = ""
last_appended_letter = None
stable_frames = 0
FRAMES_TO_APPEND = 10 # Frames needed to confirm a letter
SPACE_FRAMES_THRESHOLD = 30 # Frames of no hand to append a space

# Gemini correction
corrected_sentence = ""
is_correcting = False

def correct_grammar(raw_sentence):
    global corrected_sentence, is_correcting
    if not genai:
        corrected_sentence = "GenerativeAI not installed."
        is_correcting = False
        return
        
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are an expert sign language interpreter. "
            "Convert the following sequence of fingerspelled letters and words into a highly accurate, "
            "fluent English sentence. Fix typos and infer missing words.\n\n"
            f"Raw text: {raw_sentence}"
        )
        response = gemini_model.generate_content(prompt)
        corrected_sentence = response.text.strip()
    except Exception as e:
        corrected_sentence = f"API Error: {e}"
    is_correcting = False

cap = cv2.VideoCapture(0)
print("Webcam started. Spell words letter by letter!")

no_hand_frames = 0

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    predicted_letter = ""
    
    if results.multi_hand_landmarks:
        no_hand_frames = 0
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate bounding box for the hand
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            
        # Add padding to bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Crop, Grayscale, Resize, Normalize for CNN
        if x_max > x_min and y_max > y_min:
            hand_crop = frame[y_min:y_max, x_min:x_max]
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            normalized = resized.astype('float32') / 255.0
            reshaped = np.reshape(normalized, (1, 28, 28, 1))
            
            # Predict
            predictions = model.predict(reshaped, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]
            
            if confidence > 0.6:
                predicted_letter = alphabet[class_idx]
                cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Stabilization Logic
                if predicted_letter == last_appended_letter:
                    stable_frames += 1
                else:
                    stable_frames = 0
                    
                last_appended_letter = predicted_letter
                
                if stable_frames == FRAMES_TO_APPEND:
                    sentence += predicted_letter
                    stable_frames = 0
            else:
                last_appended_letter = None
                stable_frames = 0
    else:
        no_hand_frames += 1
        if no_hand_frames == SPACE_FRAMES_THRESHOLD:
            if len(sentence) > 0 and sentence[-1] != " ":
                sentence += " " # Add space between words
        
        last_appended_letter = None
        stable_frames = 0
        
    # UI Overlay
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Raw: {sentence}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if corrected_sentence:
        cv2.putText(frame, f"Translation: {corrected_sentence}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # Progress bar for letter commit
    if stable_frames > 0:
        progress = min(1.0, stable_frames / FRAMES_TO_APPEND)
        cv2.rectangle(frame, (0, h - 85), (int(w * progress), h - 80), (0, 255, 0), -1)
        
    cv2.imshow("Professional CNN Sign Language Predictor", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'):
        sentence = ""
        corrected_sentence = ""
    elif key == 8: # Backspace
        if len(sentence) > 0:
            sentence = sentence[:-1]
    elif key == 13: # Enter
        if len(sentence) > 0 and not is_correcting:
            is_correcting = True
            corrected_sentence = "Translating..."
            threading.Thread(target=correct_grammar, args=(sentence,), daemon=True).start()

cap.release()
cv2.destroyAllWindows()
