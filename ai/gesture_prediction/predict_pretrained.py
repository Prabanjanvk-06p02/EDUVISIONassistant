import cv2
import mediapipe as mp
import numpy as np
import os
import collections
import statistics
import urllib.request
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../models"))
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Pre-trained MediaPipe Gesture Recognizer model not found.")
    print("Downloading model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

# Temporal Smoothing Configuration
SMOOTHING_BUFFER_SIZE = 10
CONFIDENCE_THRESHOLD = 0.50 # MediaPipe's base confidence is usually good

# Initialize MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

recognizer = GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

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

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    # MediaPipe requires Timestamp in milliseconds for video mode
    timestamp_ms = int(time.time() * 1000)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Process
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    if recognition_result.gestures:
        # Get the highest confidence gesture from the first detected hand
        top_gesture = recognition_result.gestures[0][0]
        
        # Draw landmarks if available
        if recognition_result.hand_landmarks:
            from mediapipe.framework.formats import landmark_pb2
            for hand_landmarks in recognition_result.hand_landmarks:
                proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                proto_landmarks.landmark.extend([
                  landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                mp_draw.draw_landmarks(frame, proto_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Only append if high confidence and not 'None'
        if top_gesture.score >= CONFIDENCE_THRESHOLD and top_gesture.category_name != "None":
            prediction_buffer.append(top_gesture.category_name)
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
            if len(prediction_buffer) >= SMOOTHING_BUFFER_SIZE // 2 and most_common_prediction != "Unknown":
                final_predicted_gesture = most_common_prediction
            elif most_common_prediction == "Unknown":
                final_predicted_gesture = "Unknown (Low Confidence)"
        except statistics.StatisticsError:
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
    cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    sentence_text = " ".join(sentence)
    cv2.putText(frame, sentence_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Progress bar for word append
    if current_stable_gesture and stable_frames > 0 and current_stable_gesture != last_appended_gesture:
        progress = min(1.0, stable_frames / FRAMES_TO_APPEND)
        bar_width = int(frame.shape[1] * progress)
        cv2.rectangle(frame, (0, frame.shape[0] - 65), (bar_width, frame.shape[0] - 60), (0, 255, 0), -1)

    cv2.imshow("Pre-Trained Gesture Recognizer", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c'): # Clear sentence
        sentence = []
    elif key == 8: # Backspace
        if len(sentence) > 0:
            sentence.pop()
            last_appended_gesture = None if len(sentence) == 0 else sentence[-1]

cap.release()
cv2.destroyAllWindows()
recognizer.close()
