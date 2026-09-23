import cv2
import numpy as np
import os
import mediapipe as mp
import time

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data/dynamic_asl')
ACTIONS = np.array(['hello', 'yes', 'no', 'thanks'])
NO_SEQUENCES = 30     # 30 videos per action
SEQUENCE_LENGTH = 30  # 30 frames per video

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract and flatten both hands' keypoints (126 values total)"""
    keypoints = np.zeros(126) # 21 * 3 * 2 = 126
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2: break # Max 2 hands
            
            offset = hand_idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[offset + i*3] = lm.x
                keypoints[offset + i*3+1] = lm.y
                keypoints[offset + i*3+2] = lm.z
    return keypoints

def create_folders():
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def main():
    print("Setting up folders for Dynamic Sign Language Data...")
    create_folders()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for action in ACTIONS:
            print(f"\n--- Get Ready to record: {action.upper()} ---")
            for _ in range(3):
                print(f"Starting in {_}...")
                time.sleep(1)
            
            for sequence in range(NO_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1)
                    
                    # Make prediction
                    image, results = mediapipe_detection(frame, hands)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Apply collection logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000) # 2 second break between videos
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == 27: # ESC
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                        
    cap.release()
    cv2.destroyAllWindows()
    print("Data Collection Completed Successfully!")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
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

if __name__ == '__main__':
    main()
