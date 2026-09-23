from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import queue
import time
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import re

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- MODEL ----------------
model = joblib.load("../ai/models/gesture_model.pkl")

# ---------------- CAMERA (FIXED STABLE VERSION) ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- GLOBAL STATE ----------------
latest_gesture = "WAITING"
gesture_history = []

attendance = {}

teacher_text = "Listening..."
speech_buffer = []
summary_text = ""
keywords = []

text_lock = threading.Lock()

# ---------------- SPEECH OUTPUT ----------------
speech_queue = queue.Queue()

last_spoken = None
last_time = 0
cooldown = 2

# =====================================================
# TEXT TO SPEECH
# =====================================================
def speak(text):
    try:
        file = "temp.mp3"
        gTTS(text=str(text), lang='en').save(file)
        playsound(file)
        if os.path.exists(file):
            os.remove(file)
    except:
        pass

def speech_worker():
    while True:
        text = speech_queue.get()
        speak(text)

threading.Thread(target=speech_worker, daemon=True).start()

# =====================================================
# SPEECH TO TEXT (STABLE)
# =====================================================
def speech_to_text():
    global teacher_text, speech_buffer, summary_text, keywords

    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.pause_threshold = 1.2

    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                audio = r.listen(source)

            text = r.recognize_google(audio)

            if text.strip():
                teacher_text = text

                speech_buffer.append(text)
                if len(speech_buffer) > 5:
                    speech_buffer.pop(0)

                full_text = " ".join(speech_buffer)

                sentences = re.split(r'[.!?]', full_text)
                sentences = [s.strip() for s in sentences if s.strip()]
                summary_text = " ".join(sentences[-2:])

                words = re.findall(r'\w+', full_text.lower())
                stopwords = set(["the","is","am","are","i","you","a","an","and","to","in","on","for","of","it","this"])
                filtered = [w for w in words if w not in stopwords and len(w) > 2]
                keywords = list(set(filtered))[:10]

        except:
            continue

threading.Thread(target=speech_to_text, daemon=True).start()

# =====================================================
# ATTENDANCE
# =====================================================
def mark_attendance(gesture):
    if "yes" in gesture.lower():
        attendance["student_1"] = "Present"
    elif "no" in gesture.lower():
        attendance["student_1"] = "Absent"

# =====================================================
# CAMERA LOOP (FIXED)
# =====================================================
def generate_frames():
    global latest_gesture, last_spoken, last_time

    while True:
        time.sleep(0.03)  # 🔥 smooth camera

        success, frame = cap.read()

        # ---------------- CAMERA FIX ----------------
        if not success or frame is None:
            cap.release()
            time.sleep(0.5)
            cap.open(0)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        gesture = "WAITING"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                if len(features) == 63:
                    try:
                        gesture = str(model.predict(np.array(features).reshape(1, -1))[0])
                    except:
                        gesture = "ERROR"

        # ---------------- STABILITY ----------------
        gesture_history.append(gesture)
        if len(gesture_history) > 7:
            gesture_history.pop(0)

        stable_gesture = max(set(gesture_history), key=gesture_history.count)

        # ---------------- ATTENDANCE ----------------
        mark_attendance(stable_gesture)

        # ---------------- SPEECH ----------------
        now = time.time()

        if stable_gesture != "WAITING":
            if stable_gesture != last_spoken and now - last_time > cooldown:
                speech_queue.put(stable_gesture)
                last_spoken = stable_gesture
                last_time = now

        latest_gesture = stable_gesture

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, stable_gesture, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

# 🔥 FIXED STREAM TYPE (MOST IMPORTANT)
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/gesture")
def gesture():
    return jsonify({"gesture": latest_gesture})

@app.route("/teacher_text")
def teacher():
    with text_lock:
        return jsonify({"text": teacher_text})

@app.route("/summary")
def summary():
    return jsonify({"summary": summary_text})

@app.route("/keywords")
def get_keywords():
    return jsonify({"keywords": keywords})

@app.route("/attendance")
def att():
    return jsonify(attendance)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    print("🚀 System Running with Stable Camera Fix")
    app.run(debug=True, use_reloader=False)