import numpy as np
import cv2
import winsound
import os

# Get the absolute path to the XML file
xml_path = os.path.abspath('car2.xml')
face_cascade = cv2.CascadeClassifier(xml_path)

# Check if the file was loaded successfully
if face_cascade.empty():
    print(f"Error: Could not load classifier from {xml_path}")
    exit()

# Try different camera indices if the default one doesn't work
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Trying different indices.")
    for i in range(1, 10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera opened successfully at index {i}.")
            break
    else:
        print("Error: Could not open any camera.")
        exit()

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize alarm detection
    Alarmdetect = 0

    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (30, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    # Display vehicle count
    cv2.putText(img, "Vehicle Count - " + str(len(faces)), pos, font, fontScale, fontColor, lineType)

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Check if the number of detected vehicles exceeds the threshold
    if len(faces) >= 7:
        Alarmdetect = 1
        pos1 = (80, 80)
        fontColor1 = (0, 0, 255)
        cv2.putText(img, "High Road Traffic", pos1, font, fontScale + 0.3, fontColor1, lineType)

    # Sound an alarm if high road traffic is detected
    if Alarmdetect == 1:
        frequency = 1500  # Set Frequency To 1500 Hertz
        duration = 500  # Set Duration To 500 ms
        winsound.Beep(frequency, duration)

    # Display the resulting frame
    cv2.imshow('img', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
