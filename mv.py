import cv2
import numpy as np

def detect_moving_objects(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Create background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Check if the frame was successfully read
        if not ret:
            break
        
        # Apply the background subtractor to get the foreground mask
        fg_mask = bg_subtractor.apply(frame)
        
        # Remove noise from the foreground mask using opening operation
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of the detected objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding box around moving objects
        for contour in contours:
            # Filter out small contours/noise by area
            if cv2.contourArea(contour) > 1000:  # adjust area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the resulting frame with detected objects
        cv2.imshow('Moving Object Detection', frame)
        
        # Check for 'Esc' key press to exit the loop
        if cv2.waitKey(30) & 0xFF == 27:  # 30 ms delay between frames, 'Esc' key
            break
    
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_file = 'path_to_your_video_file.mp4'
detect_moving_objects(video_file)
