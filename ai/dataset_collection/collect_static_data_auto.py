"""
=============================================================
  AUTO STATIC DATA COLLECTOR
  Collects 200+ samples per gesture automatically.
  - Auto-captures frames (no need to press SPACE every time)
  - Countdown before capture starts
  - Shows live progress bar
  - Saves raw (unnormalized) keypoints - training script normalizes
  - Supports appending to existing dataset
=============================================================
  Usage:
    python collect_static_data_auto.py
  Controls during capture:
    P  - Pause / Resume auto-capture
    S  - Save single frame manually
    ESC - Stop & move to next gesture (or exit)
=============================================================
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Config ──────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))
SAMPLES_TARGET = 200          # samples to collect per gesture
CAPTURE_FPS    = 10           # auto-capture rate (frames per second)
COUNTDOWN_SEC  = 3            # countdown before capture starts

# Signs to collect — edit this list freely
SIGNS_TO_COLLECT = [
    # ── Core communication ──
    "hello", "yes", "no", "No",
    # ── Pronouns ──
    "i", "you", "we", "my", "your",
    # ── Verbs ──
    "go", "help", "want", "need", "like",
    "eat", "drink", "sleep", "stop", "come",
    # ── Nouns ──
    "food", "water", "home", "school", "name",
    "mother", "father", "friend", "doctor", "pain",
    # ── Adjectives / misc ──
    "good", "bad", "more", "please", "sorry",
    "not", "thank you", "where", "what", "how",
]

# ── MediaPipe ────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ── Helpers ──────────────────────────────────────────────────
def draw_progress_bar(frame, current, total, x=10, y=120, w=400, h=20):
    pct = min(1.0, current / max(1, total))
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
    fill_color = (0, 220, 100) if pct < 0.75 else (0, 180, 255)
    cv2.rectangle(frame, (x, y), (x + int(w*pct), y+h), fill_color, -1)
    cv2.putText(frame, f"{current}/{total}", (x+w+10, y+h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def get_existing_count(gesture_dir):
    if not os.path.exists(gesture_dir):
        return 0
    return len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])

def extract_keypoints(results):
    """Extract 126-feature vector (2 hands x 21 landmarks x 3 coords)."""
    keypoints = np.zeros(126, dtype=np.float32)
    if results.multi_hand_landmarks:
        for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break
            offset = hand_idx * 63
            for i, lm in enumerate(hand_lm.landmark):
                keypoints[offset + i*3]     = lm.x
                keypoints[offset + i*3 + 1] = lm.y
                keypoints[offset + i*3 + 2] = lm.z
    return keypoints

def collect_gesture(cap, gesture_name):
    """Collect samples for one gesture. Returns number of new samples saved."""
    gesture_dir  = os.path.join(DATASET_FOLDER, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    frame_id     = get_existing_count(gesture_dir)
    already_had  = frame_id
    need         = max(0, SAMPLES_TARGET - frame_id)

    if need == 0:
        print(f"  [{gesture_name}] Already has {frame_id} samples — skipping.")
        return 0

    print(f"\n{'='*55}")
    print(f"  Gesture : {gesture_name}")
    print(f"  Have    : {already_had}  |  Need: {need}  |  Target: {SAMPLES_TARGET}")
    print(f"  GET READY — countdown will start in the window")
    print(f"  Controls: P=Pause  S=Save manual  ESC=Skip gesture")
    print(f"{'='*55}")

    # ── Countdown ──
    countdown_end = time.time() + COUNTDOWN_SEC
    while time.time() < countdown_end:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        remaining = int(countdown_end - time.time()) + 1
        cv2.putText(frame, f"GET READY: {remaining}", (80, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 5)
        cv2.putText(frame, f"Sign: {gesture_name}", (80, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)

    # ── Auto capture loop ──
    paused       = False
    last_capture = time.time()
    interval     = 1.0 / CAPTURE_FPS
    new_saved    = 0

    while frame_id < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_mp.process(rgb)

        hand_detected = results.multi_hand_landmarks is not None
        num_hands     = len(results.multi_hand_landmarks) if hand_detected else 0

        # Draw landmarks
        if hand_detected:
            colors = [(0, 255, 128), (0, 128, 255)]
            for hidx, hlm in enumerate(results.multi_hand_landmarks):
                if hidx >= 2:
                    break
                mp_draw.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=colors[hidx], thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(180, 180, 180), thickness=1)
                )

        # HUD
        status_color = (0, 255, 100) if hand_detected else (0, 0, 255)
        status_text  = f"Hands: {num_hands}/2" if hand_detected else "NO HAND DETECTED"
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(frame, "PAUSED" if paused else "AUTO CAPTURING",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 180, 255) if paused else (0, 255, 100), 2)
        draw_progress_bar(frame, frame_id, SAMPLES_TARGET)

        # Auto-capture
        now = time.time()
        if not paused and hand_detected and (now - last_capture) >= interval:
            kp = extract_keypoints(results)
            save_path = os.path.join(gesture_dir, f"{frame_id}.npy")
            np.save(save_path, kp)
            frame_id   += 1
            new_saved  += 1
            last_capture = now
            # Flash effect
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 255, 100), 4)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC — skip gesture
            print(f"  Skipped. Saved {new_saved} new samples for '{gesture_name}'.")
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'}")
        elif key == ord('s') or key == ord('S'):
            if hand_detected:
                kp = extract_keypoints(results)
                save_path = os.path.join(gesture_dir, f"{frame_id}.npy")
                np.save(save_path, kp)
                frame_id  += 1
                new_saved += 1
                print(f"  Manual save: frame {frame_id}")

    print(f"  Done. Total for '{gesture_name}': {frame_id} samples (+{new_saved} new)")
    return new_saved

# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    print("\n" + "="*55)
    print("  SIGN LANGUAGE AUTO DATA COLLECTOR")
    print("="*55)
    print(f"  Target samples per gesture : {SAMPLES_TARGET}")
    print(f"  Auto-capture rate          : {CAPTURE_FPS} fps")
    print(f"  Signs to collect           : {len(SIGNS_TO_COLLECT)}")
    print()

    # Show current counts
    print("  Current dataset status:")
    for sign in SIGNS_TO_COLLECT:
        d     = os.path.join(DATASET_FOLDER, sign)
        count = get_existing_count(d)
        bar   = "#" * int(count / SAMPLES_TARGET * 20)
        print(f"    {sign:15s} {count:3d}/{SAMPLES_TARGET}  [{bar:<20}]")
    print()

    mode = input("  Collect ALL signs? (a) or pick ONE? (o): ").strip().lower()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if mode == "o":
        print("\n  Available signs:")
        for i, s in enumerate(SIGNS_TO_COLLECT):
            print(f"    {i:2d}. {s}")
        idx = int(input("  Enter number: ").strip())
        signs = [SIGNS_TO_COLLECT[idx]]
    else:
        signs = SIGNS_TO_COLLECT

    total_saved = 0
    for sign in signs:
        saved = collect_gesture(cap, sign)
        total_saved += saved
        # Brief pause between gestures
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Collection complete! {total_saved} new samples saved total.")
    print(f"  Dataset folder: {DATASET_FOLDER}")
    print("\n  Run train_static_model_v2.py to train the model now.\n")

if __name__ == "__main__":
    main()
