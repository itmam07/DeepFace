import cv2
from deepface import DeepFace
import threading

# 1. Define a global variable to store the latest emotion
# This will be shared between the "Video Thread" and the "AI Thread"
current_emotion = "Analyzing..."

def analyze_emotion(frame):
    """
    This function runs in a separate thread (background).
    It analyzes the frame and updates the global 'current_emotion' variable.
    """
    global current_emotion
    try:
        # We use enforce_detection=False to prevent it from crashing if no face is found
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # DeepFace returns a list in newer versions; handle both list and dict
        if isinstance(result, list):
            result = result[0]
            
        current_emotion = result['dominant_emotion']
        
    except Exception as e:
        # If something goes wrong (e.g., face lost), keep the last known emotion or reset
        print(f"Error: {e}")
        pass

# 2. Start the Webcam
cap = cv2.VideoCapture(0)

# Counter to control how often we send a frame to the AI
frame_count = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 3. The "AI Logic" (Runs every 30 frames to save resources)
    # We don't need to analyze every single millisecond.
    if frame_count % 30 == 0:
        # We send a copy of the frame to the background thread
        # 'target' is the function to run, 'args' is the input (the current frame)
        threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()

    # 4. The "Display Logic" (Runs every single frame - Smooth!)
    # Draw a black bar at the top for text visibility
    cv2.rectangle(frame, (0, 0), (300, 50), (0, 0, 0), -1)
    
    # Display the emotion text
    cv2.putText(frame, f"Emotion: {current_emotion}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Smooth Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()