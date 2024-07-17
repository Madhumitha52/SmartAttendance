import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load saved data for face recognition
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image for overlay
imgbackground = cv2.imread("Smart Attendance bg.jpg")

# Define column names for attendance CSV
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Ensure 'Attendance' directory exists or create it
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        # Prepare attendance CSV file path
        csv_file = "Attendance/Attendance_" + date + ".csv"

        # Check if file exists or create it
        exist = os.path.isfile(csv_file)

        # Draw rectangles and text on the frame for visualization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (50, 50, 255), 2)  # Corrected rectangle usage
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Prepare attendance record
        attendance = [str(output[0]), str(timestamp)]

    # Overlay the frame on the background image for display
    imgbackground[162:162 + 480, 55:55 + 640] = frame

    cv2.imshow("frame", imgbackground)
    k = cv2.waitKey(1)

    # Option to mark attendance with 'o' key
    if k == ord('o'):
        time.sleep(5)

        # Append or create attendance CSV file
        with open(csv_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    # Exit loop with 'q' key
    if k == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
