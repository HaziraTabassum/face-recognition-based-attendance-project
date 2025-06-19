import cv2
import numpy as np
import os
import pickle
from datetime import datetime


facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if facedetect.empty():
    print("Error: Haar cascade file not found. Please ensure it is in the correct path.")
    exit()

# Load face data and names
data_path = "data"
face_data_file = os.path.join(data_path, 'face_data.pkl')
names_file = os.path.join(data_path, 'names.pkl')

if not os.path.exists(face_data_file) or not os.path.exists(names_file):
    print("Error: Face data or names file not found. Run 'dataset.py' first to collect face data.")
    exit()

with open(face_data_file, 'rb') as f:
    face_data = pickle.load(f)

with open(names_file, 'rb') as f:
    names = pickle.load(f)


face_data = face_data.reshape(face_data.shape[0], -1)

# Convert to float32 for kNN
face_data = face_data.astype(np.float32)

# Initialize kNN for face recognition
knn = cv2.ml.KNearest_create()
labels = np.arange(len(names))  # Generate labels for training
knn.train(face_data, cv2.ml.ROW_SAMPLE, labels)


attendance_file = "attendance.csv"

def mark_attendance(name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # Check if attendance file exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Name,Timestamp\n")
    # Append attendance data
    with open(attendance_file, 'r+') as f:
        data = f.readlines()
        attendance_set = set(line.split(',')[0] for line in data[1:])
        if name not in attendance_set:  # Avoid duplicate entries
            f.write(f"{name},{timestamp}\n")
            print(f"Attendance marked for {name} at {timestamp}")


video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Attendance system initialized. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().astype(np.float32)

        # Perform face recognition using kNN
        _, result, _, _ = knn.findNearest(np.array([resized_img]), k=5)
        label = int(result[0].item())
        name = names[label]

        # Mark attendance
        mark_attendance(name)

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
        print("Exiting attendance system...")
        break

# Release resources
video.release()
cv2.destroyAllWindows()
print(f"Attendance saved to '{attendance_file}'.")

