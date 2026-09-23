import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("../models/gesture_model.pkl")

# ----------------------------
# TEXT TO SPEECH
# ----------------------------
engine = pyttsx3.init()

# Optional: Adjust voice speed
engine.setProperty('rate', 150)

# ----------------------------
# GESTURE MESSAGES
# ----------------------------
GESTURE_MESSAGES = {
    "hello": "Hello",
    "help": "I need help",
    "yes": "Yes",
    "no": "No",
    "water": "I need water"
}

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ----------------------------
# WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# STABILITY VARIABLES
# ----------------------------
gesture_history = []

last_spoken_gesture = ""
last_spoken_time = 0

COOLDOWN_SECONDS = 2

print("GestureVoice AI Started")
print("Press ESC to Exit")

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    predicted_gesture = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = []

            for lm in hand_landmarks.landmark:
                features.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            features = np.array(features).reshape(1, -1)

            predicted_gesture = model.predict(features)[0]

    # ----------------------------
    # STABLE PREDICTION
    # ----------------------------
    gesture_history.append(predicted_gesture)

    if len(gesture_history) > 10:
        gesture_history.pop(0)

    stable_gesture = None

    if len(gesture_history) == 10:

        if len(set(gesture_history)) == 1:

            stable_gesture = gesture_history[0]

    # ----------------------------
    # SPEAK GESTURE
    # ----------------------------
    current_time = time.time()

    if (
        stable_gesture
        and stable_gesture in GESTURE_MESSAGES
        and (
            stable_gesture != last_spoken_gesture
            or current_time - last_spoken_time > COOLDOWN_SECONDS
        )
    ):

        message = GESTURE_MESSAGES[stable_gesture]

        print(f"Speaking: {message}")

        engine.say(message)
        engine.runAndWait()

        last_spoken_gesture = stable_gesture
        last_spoken_time = current_time

    # ----------------------------
    # DISPLAY TEXT
    # ----------------------------
    cv2.putText(
        frame,
        f"Gesture: {predicted_gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if stable_gesture:

        cv2.putText(
            frame,
            f"Stable: {stable_gesture}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "GestureVoice AI - Gesture To Speech",
        frame
    )

    # ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()