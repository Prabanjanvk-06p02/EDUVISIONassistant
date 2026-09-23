import cv2
import time
import os
import threading
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    exit(1)

# Configure your Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Use the multimodal Gemini 1.5 Flash model
gemini_model = genai.GenerativeModel('gemini-flash-latest')

full_sentence = ""
is_correcting_grammar = False
status_text = "Ready. Press SPACE to start signing."

def correct_grammar(raw_sentence):
    global status_text, is_correcting_grammar, full_sentence
    try:
        prompt = (
            "You are an expert sign language interpreter. "
            "Convert the following sequence of sign language keywords/words into a rich, grammatically correct, "
            "and highly accurate natural English sentence. Infer and add missing words "
            "to ensure the sentence is complete. Only reply with the final corrected sentence and nothing else.\n\n"
            f"Raw words: {raw_sentence}"
        )
        response = gemini_model.generate_content(prompt)
        full_sentence = response.text.strip()
        status_text = "Perfect sentence formed!"
    except Exception as e:
        error_msg = f"Grammar Error: {e}"
        print(f"\n[!] {error_msg}")
        status_text = error_msg
        
    is_correcting_grammar = False

def translate_sign_language(frames):
    global status_text, is_translating, full_sentence
    
    try:
        prompt = (
            "You are an expert American Sign Language (ASL) interpreter. "
            "I am providing you with a sequence of frames from a video of someone signing. "
            "Please carefully analyze the hand gestures, movements, and body language across these frames. "
            "Translate the sign language into a highly precise, fluent, and natural English sentence. "
            "Only output the final translated sentence, with no conversational filler or explanations."
        )
        
        # Combine the prompt and the sequence of PIL Images
        inputs = [prompt] + frames
        
        response = gemini_model.generate_content(inputs)
        new_text = response.text.strip()
        
        if full_sentence == "":
            full_sentence = new_text
        else:
            full_sentence += " " + new_text
            
        status_text = "Translation complete. Ready for next word."
    except Exception as e:
        error_msg = f"Translation Error: {e}"
        print(f"\n[!] {error_msg}")
        status_text = error_msg
    
    is_translating = False


# Webcam Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting Expert Gemini ASL Vision Translator...")
print("Instructions:")
print("1. Press 'SPACE' to start recording your sign language.")
print("2. Press 'SPACE' again to stop and send the video frames to Gemini for translation.")
print("3. Press 'C' to clear the current accumulated sentence.")
print("4. Press 'ENTER' to format your accumulated words into a perfect sentence!")
print("5. Press 'ESC' to exit.")

recording = False
frames_buffer = []
is_translating = False
last_frame_time = 0
FRAME_CAPTURE_INTERVAL = 0.33 # Capture a frame every 0.33 seconds (3 FPS) to speed up upload

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 32: # SPACE
        if not recording and not is_translating:
            recording = True
            frames_buffer = []
            status_text = "Recording... Sign now! Press SPACE to stop."
            print("Recording started...")
        elif recording:
            recording = False
            if len(frames_buffer) == 0:
                status_text = "Too fast! Hold SPACE longer to capture frames."
            else:
                status_text = "Translating... Please wait."
                is_translating = True
                print(f"Recording stopped. Sent {len(frames_buffer)} frames to Gemini.")
                threading.Thread(target=translate_sign_language, args=(frames_buffer,), daemon=True).start()
            
    elif key == 27: # ESC
        break
    elif key == ord('c') or key == ord('C'):
        full_sentence = ""
        status_text = "Sentence cleared. Ready. Press SPACE to start signing."
    elif key == 13: # Enter key
        if full_sentence and not is_correcting_grammar and not is_translating:
            is_correcting_grammar = True
            status_text = "Forming perfect sentence... Please wait."
            threading.Thread(target=correct_grammar, args=(full_sentence,), daemon=True).start()

    # If recording, capture frames periodically
    if recording:
        current_time = time.time()
        if current_time - last_frame_time >= FRAME_CAPTURE_INTERVAL:
            # Convert CV2 frame (BGR) to PIL Image (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Massive speed optimization: shrink images to 256x256
            pil_img.thumbnail((256, 256)) 
            
            frames_buffer.append(pil_img)
            last_frame_time = current_time
            
            # Visual feedback that a frame was captured
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)

    # UI Overlays
    cv2.rectangle(display_frame, (0, display_frame.shape[0] - 80), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
    
    color = (0, 255, 0)
    if recording:
        color = (0, 0, 255)
    elif is_translating:
        color = (0, 255, 255)
        
    # Draw Status
    cv2.putText(display_frame, f"Status: {status_text}", (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw Sentence
    cv2.putText(display_frame, f"Sentence: {full_sentence}", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if recording:
        cv2.putText(display_frame, f"Frames: {len(frames_buffer)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Expert Gemini Sign Language Translator", display_frame)

cap.release()
cv2.destroyAllWindows()
