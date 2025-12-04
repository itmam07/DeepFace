# DeepFace Emotion Recognition

Real-time emotion detection using facial recognition powered by DeepFace and OpenCV.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Scripts

### `recognition.py`
**Simple streaming mode** - Minimal implementation using DeepFace's built-in streaming functionality. Opens webcam and analyzes emotions with face recognition database support.

### `smooth-recognition.py`
**Optimized console mode** - Uses threading to separate AI analysis from video display. Analyzes every 30th frame to reduce CPU usage while maintaining smooth 60fps video feed. Press 'q' to quit.

### `recognition-ui.py`
**Modern GUI mode** - Full-featured desktop application with CustomTkinter interface. Features:
- Clean dark-themed UI
- Real-time emotion display with color-coded feedback
- Threaded processing for smooth performance
- 640x480 video preview

## Usage

Run any script directly:
```bash
python recognition.py
python smooth-recognition.py
python recognition-ui.py
```

## Detected Emotions

- Happy
- Sad
- Angry
- Neutral
- Fear
- Disgust
- Surprise

## Requirements

- Python 3.7+
- Webcam access
- See `requirements.txt` for package dependencies
