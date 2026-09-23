import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import collections
import threading

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../models/dynamic_lstm_model.keras')
ACTIONS = np.array(['hello', 'yes', 'no', 'thanks'])
SEQUENCE_LENGTH = 30  # 30 frames per video

try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if genai:
    genai.configure(api_key=GEMINI_API_KEY)

# Load LSTM Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("LSTM Dynamic Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run collect_dynamic_data.py and train_dynamic_model.py first.")
    exit(1)

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    keypoints = np.zeros(126) 
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2: break 
            
            offset = hand_idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[offset + i*3] = lm.x
                keypoints[offset + i*3+1] = lm.y
                keypoints[offset + i*3+2] = lm.z
    return keypoints

def mediapipe_detection(image, mp_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = mp_model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

# Sentence and Grammar state
sentence = []
corrected_sentence = ""
is_correcting = False

def correct_grammar(raw_sentence):
    global corrected_sentence, is_correcting
    if not genai:
        corrected_sentence = "Error: google-generativeai not installed."
        is_correcting = False
        return
        
    try:
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        prompt = (
            "You are an expert sign language interpreter. "
            "Convert the following sequence of sign language keywords into a rich, grammatically correct, "
            "and highly accurate natural English sentence. Infer and add missing words "
            "to ensure the sentence is complete. Only reply with the final corrected sentence and nothing else.\n\n"
            f"Raw words: {raw_sentence}"
        )
        response = gemini_model.generate_content(prompt)
        corrected_sentence = response.text.strip()
    except Exception as e:
        corrected_sentence = f"API Error: {e}"
    is_correcting = False

def main():
    global sentence, corrected_sentence, is_correcting
    
    sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
    predictions = []
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("--- Dynamic Predictor Running ---")
    print("Press ESC to exit.")
    print("Press C to clear sentence.")
    print("Press ENTER to form grammar sentence.")
    
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            # Predict once we have a full window
            if len(sequence) == SEQUENCE_LENGTH:
                # The model expects a batch shape: (1, 30, 126)
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                predicted_action = ACTIONS[np.argmax(res)]
                confidence = res[np.argmax(res)]
                predictions.append(np.argmax(res))
                
                # We only count it as a valid prediction if it is stable over the last 10 frames
                # and confidence is very high
                if np.unique(predictions[-10:])[0] == np.argmax(res) and confidence > 0.90:
                    if len(sentence) > 0: 
                        if ACTIONS[np.argmax(res)] != sentence[-1]:
                            sentence.append(ACTIONS[np.argmax(res)])
                    else:
                        sentence.append(ACTIONS[np.argmax(res)])

            # UI Overlays
            cv2.rectangle(image, (0, image.shape[0] - 100), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            
            # Draw raw sentence
            cv2.putText(image, "Raw: " + ' '.join(sentence), (10, image.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Draw Corrected sentence
            if corrected_sentence:
                cv2.putText(image, "Output: " + corrected_sentence, (10, image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Instant LSTM Prediction', image)
            
            # Key handlers
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC
                break
            elif key == ord('c') or key == ord('C'):
                sentence = []
                corrected_sentence = ""
                sequence.clear() # clear rolling buffer
                predictions = []
            elif key == 13: # ENTER
                if len(sentence) > 0 and not is_correcting:
                    is_correcting = True
                    corrected_sentence = "Correcting grammar..."
                    raw_text = " ".join(sentence)
                    threading.Thread(target=correct_grammar, args=(raw_text,), daemon=True).start()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
