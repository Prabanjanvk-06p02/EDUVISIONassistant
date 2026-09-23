import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import collections
import statistics
import threading

try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if genai and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/static_gesture_model.keras"))
CLASSES_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../models/static_class_names.pkl"))

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

# Temporal Smoothing Configuration
SMOOTHING_BUFFER_SIZE = 10
CONFIDENCE_THRESHOLD = 0.90 # Must be 90% confident

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = joblib.load(CLASSES_PATH)
    print("Highly Accurate Static Keypoint Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run collect_static_data.py and train_static_model.py first.")
    exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam at index 0.")
    exit(1)

print("Webcam successfully opened! Press ESC to exit.")

# Buffer for temporal smoothing
prediction_buffer = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)
final_predicted_gesture = "No Hand Detected"

# Sentence formation variables
sentence = []
last_appended_gesture = None
current_stable_gesture = None
stable_frames = 0
FRAMES_TO_APPEND = 15

# Grammar correction state
corrected_sentence = ""
is_correcting = False

def correct_grammar(raw_sentence):
    global corrected_sentence, is_correcting
    if not genai:
        corrected_sentence = "Error: google-generativeai not installed."
        is_correcting = False
        return
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        corrected_sentence = "Error: GEMINI_API_KEY not set in script."
        is_correcting = False
        return
        
    try:
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        prompt = (
            "You are an expert sign language interpreter. "
            "Convert the following sequence of sign language keywords into a rich, grammatically correct, "
            "and highly accurate natural English sentence. Infer and add missing words "
            "(such as pronouns, articles, verbs, and prepositions) to ensure the sentence is complete and makes sense in context. "
            "Your output must be error-free. Only reply with the final corrected sentence and nothing else.\n\n"
            f"Raw signs: {raw_sentence}"
        )
        response = gemini_model.generate_content(prompt)
        corrected_sentence = response.text.strip()
    except Exception as e:
        corrected_sentence = f"API Error: {e}"
    is_correcting = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        keypoints = np.zeros(126)
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break
                
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract keypoints
            offset = hand_idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[offset + i*3] = lm.x
                keypoints[offset + i*3+1] = lm.y
                keypoints[offset + i*3+2] = lm.z
        
        # Normalize the keypoints
        normalized_keypoints = normalize_keypoints(keypoints)
        
        # Predict single frame
        res = model.predict(np.expand_dims(normalized_keypoints, axis=0), verbose=0)[0]
        class_idx = np.argmax(res)
        confidence = res[class_idx]
        
        # Only append if high confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_buffer.append(class_names[class_idx])
        else:
            prediction_buffer.append("Unknown")
    else:
        # Hand left frame, clear buffer
        prediction_buffer.clear()
        final_predicted_gesture = "No Hand Detected"

    # Temporal Smoothing logic (Majority Vote)
    if len(prediction_buffer) > 0:
        try:
            # Find the most common prediction in the buffer
            most_common_prediction = statistics.mode(prediction_buffer)
            # Only update if the buffer is somewhat full to prevent immediate flickering
            if len(prediction_buffer) >= SMOOTHING_BUFFER_SIZE // 2 and most_common_prediction != "Unknown":
                final_predicted_gesture = most_common_prediction
            elif most_common_prediction == "Unknown":
                final_predicted_gesture = "Unknown (Low Confidence)"
        except statistics.StatisticsError:
            # Tie in the mode calculation, wait for more frames
            pass

    # Sentence formation logic
    if final_predicted_gesture not in ["No Hand Detected", "Unknown (Low Confidence)"]:
        if final_predicted_gesture == current_stable_gesture:
            stable_frames += 1
            if stable_frames == FRAMES_TO_APPEND:
                if final_predicted_gesture != last_appended_gesture:
                    sentence.append(final_predicted_gesture)
                    last_appended_gesture = final_predicted_gesture
        else:
            current_stable_gesture = final_predicted_gesture
            stable_frames = 1
    elif final_predicted_gesture == "No Hand Detected":
        last_appended_gesture = None
        current_stable_gesture = None
        stable_frames = 0
    else:
        current_stable_gesture = None
        stable_frames = 0

    # UI Overlay
    color = (0, 255, 0) if final_predicted_gesture not in ["No Hand Detected", "Unknown (Low Confidence)"] else (0, 0, 255)
    cv2.putText(
        frame,
        f"Gesture: {final_predicted_gesture}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    # Sentence Overlay
    cv2.rectangle(frame, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    
    # Raw sentence
    sentence_text = "Raw: " + " ".join(sentence)
    cv2.putText(frame, sentence_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Corrected sentence
    if corrected_sentence:
        cv2.putText(frame, f"Output: {corrected_sentence}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Progress bar for word append
    if current_stable_gesture and stable_frames > 0 and current_stable_gesture != last_appended_gesture:
        progress = min(1.0, stable_frames / FRAMES_TO_APPEND)
        bar_width = int(frame.shape[1] * progress)
        cv2.rectangle(frame, (0, frame.shape[0] - 105), (bar_width, frame.shape[0] - 100), (0, 255, 0), -1)

    cv2.imshow("High Accuracy Gesture Recognition", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'): # Clear sentence
        sentence = []
        corrected_sentence = ""
    elif key == 8: # Backspace
        if len(sentence) > 0:
            sentence.pop()
            last_appended_gesture = None if len(sentence) == 0 else sentence[-1]
            corrected_sentence = ""
    elif key == 13: # Enter
        if len(sentence) > 0 and not is_correcting:
            is_correcting = True
            corrected_sentence = "Correcting grammar..."
            raw_text = " ".join(sentence)
            threading.Thread(target=correct_grammar, args=(raw_text,), daemon=True).start()

cap.release()
cv2.destroyAllWindows()