import numpy as np
import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for faster detection
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert to grayscale for faster detection
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)

    # Detect people in the image
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    # Draw bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the output video
    out.write(frame_resized)

    # Display the resulting frame
    cv2.imshow('frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

