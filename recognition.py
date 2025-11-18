from deepface import DeepFace

# This single line opens the webcam and analyzes emotions in real-time
DeepFace.stream(db_path="/home/itmam/tmp", time_threshold=1, frame_threshold=1)