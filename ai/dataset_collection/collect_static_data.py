import cv2
import mediapipe as mp
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.normpath(os.path.join(BASE_DIR, "../datasets/Custom_Data"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # Now supporting 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def collect_data():
    gesture_name = input("Enter the name of the static gesture to record (e.g., 'A', 'B', 'C'): ").strip().lower()
    
    if not gesture_name:
        print("Invalid name. Exiting.")
        return
        
    gesture_dir = os.path.join(DATASET_FOLDER, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    
    existing_files = os.listdir(gesture_dir)
    frame_id = len(existing_files)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
        
    print(f"\nRecording static frames for gesture: '{gesture_name}'")
    print("Instructions:")
    print("1. Hold your hand in the gesture position.")
    print("2. Press the 'SPACE' bar to save ONE frame.")
    print("3. Move your hand slightly or change angles, then press 'SPACE' again.")
    print("4. Press 'ESC' to exit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        cv2.putText(frame, f"Gesture: {gesture_name} | Saved Frames: {frame_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "Press SPACE to Save Frame, ESC to Exit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        keypoints = None
        if results.multi_hand_landmarks:
            keypoints = np.zeros(126) # 21*3*2
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Stop if we found more than 2 hands (shouldn't happen with max_num_hands=2)
                if hand_idx >= 2:
                    break
                    
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                offset = hand_idx * 63
                for i, lm in enumerate(hand_landmarks.landmark):
                    keypoints[offset + i*3] = lm.x
                    keypoints[offset + i*3+1] = lm.y
                    keypoints[offset + i*3+2] = lm.z

        cv2.imshow("Static Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == 32: # SPACE
            if keypoints is not None:
                save_path = os.path.join(gesture_dir, f"{frame_id}.npy")
                np.save(save_path, keypoints)
                print(f"Saved frame {frame_id} for '{gesture_name}'.")
                frame_id += 1
            else:
                print("No hand detected! Frame not saved.")
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    collect_data()
