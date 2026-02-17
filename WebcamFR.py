"""
Author: Daniel Gonzalez
Project: Biometric Face Detection (Spring 2026)
Description: This script uses OpenCV's Haar Cascade to perform 
             real-time face detection via webcam.
"""

import cv2

# Set the classifier path
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Try index 0 first; if it fails, try 1
video_capture = cv2.VideoCapture(0)

print("Starting live stream... Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    
    # Safety check: if the camera isn't ready, skip this frame
    if not ret or frame is None:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    # Complexity Logic: Show status based on whether a face is found
    if len(faces) == 0:
        cv2.putText(frame, "Status: No Match", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Status: {len(faces)} Face(s) Found", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the green boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the window
    cv2.imshow('Video Face Detection', frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
