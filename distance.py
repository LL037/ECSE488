import cv2

def calculate_distance(focal_length, real_height, image_height):
    # Simple distance formula
    if image_height == 0:
        return float('inf')  # Avoid division by zero
    return (focal_length * real_height) / image_height

def main():
    # Camera parameters and object details
    KNOWN_HEIGHT = 1.0  # Known height of the object in meters (e.g., height of a person)
    FOCAL_LENGTH = 800  # Example focal length in pixels (this should be calibrated for your camera and setup)

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame captured from the video.")
            break

        # Apply background subtraction
        fg_mask = backSub.apply(frame)

        # Threshold the mask to remove shadows and other noise
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # Define the kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Perform opening to remove noise
        mask_cleaned = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        # Find contours in the cleaned mask
        contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum contour area to consider for moving objects
        min_contour_area = 40000
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Draw bounding boxes around moving objects and calculate distance
        for contour in large_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            distance = calculate_distance(FOCAL_LENGTH, KNOWN_HEIGHT, h)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Detected Objects', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
