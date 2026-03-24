## Student Attention System: What I Have Learned So Far

This document is a simple summary of my current learning.
It explains the basic tools, how face detection works, and important practical points.

## 1. Required Language and Libraries

### Language
- Python

### Main Libraries
1. OpenCV (`cv2`)
- Open-source computer vision library.
- Used to:
    - Open webcam
    - Capture image/video frames
    - Convert color formats (BGR to RGB or grayscale)
    - Draw shapes/text on frames
    - Show live video window

2. MediaPipe (`mediapipe`)
- Google library for face/hand/body landmarks.
- Used for detailed face landmark detection.

## 2. Two Face Detection Approaches

### A. Haar Cascade (Classic/OpenCV Method)
- Old but simple and fast for basic face detection.
- It answers: "Is there a face here or not?"
- Uses a pre-trained XML model.
- Works by sliding a small window across the image at different scales.
- Best for:
    - Basic face presence detection
    - Lightweight projects

Limitations:
- Not very accurate for side faces, poor light, or complex scenes.
- Gives face box, not detailed landmarks.

### B. MediaPipe Face Mesh (Modern Method)
- More advanced and gives detailed face structure.
- Can detect many landmark points on face (commonly 468 points).
- Useful for:
    - Attention tracking
    - Head pose direction
    - Expression/face movement features

Limitations:
- Slightly heavier than Haar Cascade.
- API/version compatibility can cause errors if code and package version do not match.

## 3. Basic Real-Time Pipeline (Camera to Output)

1. Open camera with `cv2.VideoCapture(0)`.
2. Read each frame using `ret, frame = cap.read()`.
3. Check `ret`:
     - If false, camera frame capture failed.
4. Process frame:
     - Haar Cascade: convert to grayscale.
     - MediaPipe: convert BGR to RGB.
5. Detect face/landmarks.
6. Draw results (rectangle, point, text).
7. Show output with `cv2.imshow()`.
8. Exit on ESC key.
9. Release resources after loop:
     - `cap.release()`
     - `cv2.destroyAllWindows()`

## 4. Important Beginner Mistakes (Very Useful)

1. Releasing camera inside loop
- If `cap.release()` is inside `while` loop, camera closes quickly.
- Correct: release camera after loop ends.

2. Not checking camera open status
- Always verify `cap.isOpened()` before reading frames.

3. MediaPipe API mismatch
- Some versions do not support `mp.solutions`.
- In that case, either:
    - Use compatible older version, or
    - Migrate code to MediaPipe Tasks API.

4. Wrong camera index
- If `0` fails, try `1` (especially on systems with multiple cameras).

## 5. Short Comparison

- Haar Cascade:
    - Simple, fast, basic face/no-face output.
- MediaPipe:
    - Rich face landmarks, better for attention analysis.

## 6. Current Understanding (One-Line Summary)

OpenCV handles camera and frame display, Haar Cascade gives basic face detection, and MediaPipe gives detailed face landmarks for attention-based logic.
