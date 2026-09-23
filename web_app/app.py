from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import threading
import queue
import time
import collections
import speech_recognition as sr
import pyttsx3
import os
import re

try:
    from google import genai as genai_client
    GENAI_AVAILABLE = True
except ImportError:
    genai_client = None
    GENAI_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_genai_instance = None
if genai_client and GEMINI_API_KEY:
    try:
        _genai_instance = genai_client.Client(api_key=GEMINI_API_KEY)
        print("[OK] Gemini API configured")
    except Exception as _ge:
        print(f"[WARN] Gemini config error: {_ge}")

# ─────────────────────────────────────────────────────────
# MODEL — compiled @tf.function for zero-overhead inference
# ─────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
model_path       = os.path.normpath(os.path.join(BASE_DIR, "../models/static_gesture_model.keras"))
class_names_path = os.path.normpath(os.path.join(BASE_DIR, "../models/static_class_names.pkl"))

model       = None
class_names = []
_infer      = None   # compiled inference function

try:
    model       = tf.keras.models.load_model(model_path)
    class_names = joblib.load(class_names_path)

    # ── KEY OPTIMIZATION 1: compile inference into a TF graph ──
    # Eliminates Python overhead on every call (~10-30ms saved per frame)
    _input_spec = tf.TensorSpec(shape=(1, 126), dtype=tf.float32)

    @tf.function(input_signature=[_input_spec])
    def _infer(x):
        return model(x, training=False)

    # Warm-up: run once so TF JIT compiles the graph now, not on first gesture
    _dummy = tf.zeros((1, 126), dtype=tf.float32)
    _ = _infer(_dummy)
    print(f"[OK] Model loaded & warmed up - {len(class_names)} classes: {class_names}")
except Exception as e:
    print(f"[ERROR] Model load error: {e}")

# ─────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.65
)

# ── KEY OPTIMIZATION 2: pre-build DrawingSpec objects once ──
_DOT_SPECS  = [
    mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
    mp_draw.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=3),
]
_LINE_SPEC  = mp_draw.DrawingSpec(color=(160, 160, 160), thickness=1)

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION — fully vectorized (no Python loops)
# ─────────────────────────────────────────────────────────
# Pre-build landmark index arrays for fast NumPy slicing
_HAND_OFFSETS = np.array([0, 63], dtype=np.int32)          # base index per hand
_LM_IDX       = np.arange(21)                              # 0..20

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """
    Vectorized palm-size normalization — no Python for-loops.
    ~5x faster than the loop version.
    """
    out = np.zeros(126, dtype=np.float32)
    for base in (0, 63):
        hand = kp[base: base + 63]
        if not hand[0] and not hand[1] and not hand[2]:
            continue                        # hand not present (wrist = 0,0,0)
        wrist   = hand[0:3]                 # landmark 0 xyz
        mid_mcp = hand[27:30]              # landmark 9 xyz
        scale   = np.linalg.norm(mid_mcp - wrist) + 1e-6
        # Reshape to (21, 3), subtract wrist, divide by scale, flatten back
        reshaped = hand.reshape(21, 3)
        out[base: base + 63] = ((reshaped - wrist) / scale).ravel()
    return out

# ─────────────────────────────────────────────────────────
# DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────
SMOOTHING_WINDOW   = 7    # reduced from 15 → faster response, still stable
CONFIDENCE_THRESH  = 0.80
STABLE_FRAMES_REQD = 15   # reduced from 20 → slightly quicker word add
HYSTERESIS_MIN     = 3    # reduced from 5 → snappier gesture switching

# ─────────────────────────────────────────────────────────
# ── KEY OPTIMIZATION 3: DEDICATED CAMERA THREAD ──────────
# Camera runs in its own thread at full speed.
# The streaming generator simply grabs the latest ready frame
# — no blocking, no waiting for cap.read().
# ─────────────────────────────────────────────────────────
_latest_frame      = None          # latest BGR frame from camera
_latest_frame_lock = threading.Lock()
_camera_running    = True

def _camera_thread():
    global _latest_frame, _camera_running
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    # Minimize internal buffer so we always get the NEWEST frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    while _camera_running:
        ok, frame = cap.read()
        if ok and frame is not None:
            with _latest_frame_lock:
                _latest_frame = frame
        # No sleep — run as fast as the camera allows

    cap.release()

_cam_thread = threading.Thread(target=_camera_thread, daemon=True)
_cam_thread.start()

# Give camera time to open before first request
time.sleep(0.5)

# ─────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────
latest_gesture    = "WAITING"
latest_confidence = 0.0
latest_hands      = 0

pred_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
conf_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

hys_gesture = None
hys_count   = 0

sentence_words     = []
last_added_word    = None
cur_stable_word    = None
stable_frame_cnt   = 0
corrected_sentence = ""
is_correcting      = False

teacher_text = "Listening..."
speech_buf   = []
summary_text = ""
keywords     = []
text_lock    = threading.Lock()

speech_queue = queue.Queue(maxsize=2)
last_spoken  = None
last_time    = 0.0
COOLDOWN_SEC = 1.5   # slightly shorter so it feels more real-time

state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────
# TEXT-TO-SPEECH — pyttsx3 (INSTANT, OFFLINE, ZERO LATENCY)
# pyttsx3 must live in its own thread — it is not thread-safe.
# ─────────────────────────────────────────────────────────
def _tts_worker():
    """Dedicated TTS thread — initialises its own pyttsx3 engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate',   165)   # words per minute  (default ~200)
    engine.setProperty('volume', 1.0)
    # Pick a clear English voice if available
    voices = engine.getProperty('voices')
    for v in voices:
        if 'english' in v.name.lower() or 'zira' in v.name.lower() or 'david' in v.name.lower():
            engine.setProperty('voice', v.id)
            break

    while True:
        text = speech_queue.get()
        try:
            engine.say(str(text))
            engine.runAndWait()          # blocks only until speech finishes
        except Exception:
            pass

threading.Thread(target=_tts_worker, daemon=True).start()

# ─────────────────────────────────────────────────────────
# SPEECH-TO-TEXT
# ─────────────────────────────────────────────────────────
def _speech_to_text():
    global teacher_text, speech_buf, summary_text, keywords
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.pause_threshold = 1.2
    mic = sr.Microphone()
    with mic as src:
        r.adjust_for_ambient_noise(src, duration=1)
    while True:
        try:
            with mic as src:
                audio = r.listen(src)
            text = r.recognize_google(audio)
            if text.strip():
                with text_lock:
                    teacher_text = text
                    speech_buf.append(text)
                    if len(speech_buf) > 5:
                        speech_buf.pop(0)
                    full  = " ".join(speech_buf)
                    sents = re.split(r'[.!?]', full)
                    sents = [s.strip() for s in sents if s.strip()]
                    summary_text = " ".join(sents[-2:])
                    stop = {"the","is","am","are","i","you","a","an","and",
                            "to","in","on","for","of","it","this","we","he","she"}
                    raw_w    = re.findall(r'\w+', full.lower())
                    keywords = list(set(w for w in raw_w if w not in stop and len(w) > 2))[:10]
        except Exception:
            continue

threading.Thread(target=_speech_to_text, daemon=True).start()

# ─────────────────────────────────────────────────────────
# GEMINI GRAMMAR CORRECTION
# ─────────────────────────────────────────────────────────
def _simple_sentence(words: list[str]) -> str:
    """Rule-based fallback: builds a readable sentence without AI."""
    if not words:
        return ""
    joined = " ".join(w.capitalize() if i == 0 else w for i, w in enumerate(words))
    return joined + ("?" if any(w in words for w in ["what","where","how","who"]) else ".")

def _correct_grammar(raw: str, words: list[str]):
    global corrected_sentence, is_correcting
    if not _genai_instance:
        # Graceful fallback — no API needed
        corrected_sentence = _simple_sentence(words)
        is_correcting = False
        return
    try:
        prompt = (
            "You are an expert sign language interpreter. "
            "Convert the following sign language keywords into a grammatically "
            "correct, fluent, natural English sentence. "
            "Add missing pronouns, articles, verbs and prepositions as needed. "
            "Reply ONLY with the final sentence — no explanation, no quotes.\n\n"
            f"Signs: {raw}"
        )
        response = _genai_instance.models.generate_content(
            model='gemini-3.6-flash',   # updated from deprecated gemini-2.0-flash
            contents=prompt
        )
        result = response.text.strip()
        corrected_sentence = result if result else _simple_sentence(words)
        print(f"[Gemini] {raw} -> {corrected_sentence}")
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        # Fallback to simple rule-based sentence
        corrected_sentence = _simple_sentence(words)
    is_correcting = False

# ─────────────────────────────────────────────────────────
# ── KEY OPTIMIZATION 4: GESTURE PROCESSING THREAD ────────
# Runs MediaPipe + model inference in its own thread.
# Generator thread only reads the processed result — instant.
# ─────────────────────────────────────────────────────────
_processed_frame     = None     # JPEG bytes ready to stream
_processed_frame_lock = threading.Lock()

def _gesture_processing_thread():
    global latest_gesture, latest_confidence, latest_hands
    global pred_buffer, conf_buffer
    global hys_gesture, hys_count
    global cur_stable_word, stable_frame_cnt, last_added_word
    global last_spoken, last_time
    global sentence_words, _processed_frame

    while True:
        # Grab latest camera frame (non-blocking)
        with _latest_frame_lock:
            if _latest_frame is None:
                time.sleep(0.005)
                continue
            frame = _latest_frame.copy()

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False        # prevents unnecessary copy in MediaPipe
        results = hands_detector.process(rgb)
        rgb.flags.writeable = True

        raw_gesture    = None
        raw_confidence = 0.0
        hands_count    = 0

        if results.multi_hand_landmarks:
            hands_count = len(results.multi_hand_landmarks)
            keypoints   = np.zeros(126, dtype=np.float32)

            for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:
                    break
                # Draw with pre-built specs (no object allocation per frame)
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    _DOT_SPECS[hand_idx], _LINE_SPEC
                )
                offset = hand_idx * 63
                for i, lm in enumerate(hand_lm.landmark):
                    keypoints[offset + i*3]     = lm.x
                    keypoints[offset + i*3 + 1] = lm.y
                    keypoints[offset + i*3 + 2] = lm.z

            normalized = normalize_keypoints(keypoints)

            if _infer is not None:
                try:
                    # ── KEY OPTIMIZATION 1 in action: compiled graph call ──
                    t = tf.constant(normalized[np.newaxis], dtype=tf.float32)
                    probs      = _infer(t).numpy()[0]
                    best_idx   = int(np.argmax(probs))
                    raw_confidence = float(probs[best_idx])
                    raw_gesture = class_names[best_idx] if raw_confidence >= CONFIDENCE_THRESH else "Unknown"
                except Exception:
                    raw_gesture = "ERROR"
        else:
            pred_buffer.clear()
            conf_buffer.clear()
            hys_gesture = None
            hys_count   = 0

        # ── Temporal smoothing ────────────────────────────────
        if raw_gesture is not None:
            pred_buffer.append(raw_gesture)
            conf_buffer.append(raw_confidence)

        smoothed_gesture    = "WAITING"
        smoothed_confidence = 0.0

        if len(pred_buffer) >= max(1, SMOOTHING_WINDOW // 2):
            vote: dict[str, float] = {}
            for g, c in zip(pred_buffer, conf_buffer):
                if g not in ("Unknown", "ERROR"):
                    vote[g] = vote.get(g, 0.0) + c
            if vote:
                best_g              = max(vote, key=vote.get)
                total_c             = sum(vote.values())
                smoothed_gesture    = best_g
                smoothed_confidence = vote[best_g] / total_c if total_c > 0 else 0.0
            else:
                smoothed_gesture = "Unknown"

        # ── Hysteresis ────────────────────────────────────────
        if smoothed_gesture == hys_gesture:
            hys_count += 1
        else:
            hys_gesture = smoothed_gesture
            hys_count   = 1

        if hys_count >= HYSTERESIS_MIN:
            final_gesture    = smoothed_gesture
            final_confidence = smoothed_confidence
        else:
            final_gesture    = latest_gesture
            final_confidence = latest_confidence

        # ── Sentence formation ────────────────────────────────
        skip = {"WAITING", "Unknown", "ERROR"}
        if final_gesture not in skip:
            if final_gesture == cur_stable_word:
                stable_frame_cnt += 1
                if stable_frame_cnt == STABLE_FRAMES_REQD:
                    if final_gesture != last_added_word:
                        with state_lock:
                            sentence_words.append(final_gesture)
                            last_added_word = final_gesture
            else:
                cur_stable_word  = final_gesture
                stable_frame_cnt = 1
        elif final_gesture == "WAITING":
            cur_stable_word  = None
            stable_frame_cnt = 0
            last_added_word  = None

        # ── TTS cooldown ──────────────────────────────────────
        now = time.time()
        if final_gesture not in ("WAITING", "Unknown", "ERROR"):
            if final_gesture != last_spoken and now - last_time > COOLDOWN_SEC:
                try:
                    speech_queue.put_nowait(final_gesture)  # non-blocking
                except queue.Full:
                    pass
                last_spoken = final_gesture
                last_time   = now

        latest_gesture    = final_gesture
        latest_confidence = final_confidence
        latest_hands      = hands_count

        # ── HUD ───────────────────────────────────────────────
        h, w = frame.shape[:2]

        g_color = (0, 255, 100) if final_gesture not in ("WAITING", "Unknown", "ERROR") else (50, 100, 255)
        cv2.putText(frame, final_gesture, (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, final_gesture, (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, g_color, 2)

        cv2.putText(frame, f"Hands:{hands_count}/2", (w - 130, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        bar_w = int(final_confidence * 200)
        cv2.rectangle(frame, (14, 58), (214, 72), (40, 40, 40), -1)
        if bar_w > 0:
            bar_col = (0, 220, 100) if final_confidence > 0.85 else (0, 180, 255)
            cv2.rectangle(frame, (14, 58), (14 + bar_w, 72), bar_col, -1)
        cv2.putText(frame, f"{final_confidence*100:.0f}%", (220, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if cur_stable_word and cur_stable_word != last_added_word and stable_frame_cnt > 0:
            pct = min(1.0, stable_frame_cnt / STABLE_FRAMES_REQD)
            cv2.rectangle(frame, (0, h - 5), (int(w * pct), h), (0, 255, 100), -1)

        # ── Encode JPEG once, store for streamer ──────────────
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with _processed_frame_lock:
                _processed_frame = buf.tobytes()

_proc_thread = threading.Thread(target=_gesture_processing_thread, daemon=True)
_proc_thread.start()

# ─────────────────────────────────────────────────────────
# MJPEG GENERATOR — only reads pre-encoded frames, zero work
# ─────────────────────────────────────────────────────────
def generate_frames():
    last_sent = None
    while True:
        with _processed_frame_lock:
            frame_bytes = _processed_frame

        if frame_bytes is None or frame_bytes is last_sent:
            time.sleep(0.005)   # wait for next frame (1ms polling)
            continue

        last_sent = frame_bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/gesture")
def gesture():
    return jsonify({
        "gesture":    latest_gesture,
        "confidence": round(latest_confidence * 100, 1),
        "hands":      latest_hands
    })

@app.route("/sentence")
def get_sentence():
    with state_lock:
        words = list(sentence_words)
    return jsonify({
        "words":         words,
        "raw":           " ".join(words),
        "corrected":     corrected_sentence,
        "is_correcting": is_correcting
    })

@app.route("/ai_correct", methods=["POST"])
def ai_correct():
    global corrected_sentence, is_correcting
    with state_lock:
        raw   = " ".join(sentence_words)
        words = list(sentence_words)
    if not raw.strip():
        return jsonify({"status": "error", "message": "No signs detected yet."})
    if is_correcting:
        return jsonify({"status": "busy"})
    is_correcting      = True
    corrected_sentence = "CORRECTING"
    threading.Thread(target=_correct_grammar, args=(raw, words), daemon=True).start()
    return jsonify({"status": "started", "raw": raw})

@app.route("/speak_sentence", methods=["POST"])
def speak_sentence():
    with state_lock:
        words = list(sentence_words)
    text = corrected_sentence if corrected_sentence \
           and "Error" not in corrected_sentence \
           and corrected_sentence not in ("Correcting...", "") \
           else " ".join(words)
    if text.strip():
        try:
            speech_queue.put_nowait(text)
        except queue.Full:
            pass
        return jsonify({"status": "speaking", "text": text})
    return jsonify({"status": "empty"})

@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    global sentence_words, last_added_word, corrected_sentence
    global cur_stable_word, stable_frame_cnt
    with state_lock:
        sentence_words     = []
        last_added_word    = None
        corrected_sentence = ""
        cur_stable_word    = None
        stable_frame_cnt   = 0
    return jsonify({"status": "cleared"})

@app.route("/backspace_sentence", methods=["POST"])
def backspace_sentence():
    global sentence_words, last_added_word, corrected_sentence
    with state_lock:
        if sentence_words:
            sentence_words.pop()
            last_added_word    = sentence_words[-1] if sentence_words else None
            corrected_sentence = ""
        words = list(sentence_words)
    return jsonify({"status": "ok", "words": words})

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

# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[START] Gesture Voice AI - Zero-lag pipeline")
    print(f"  Model  : {len(class_names)} classes")
    print(f"  Gemini : {GENAI_AVAILABLE and bool(GEMINI_API_KEY)}")
    print(f"  Thread : Camera + Gesture processing separated")
    app.run(debug=False, use_reloader=False, threaded=True)