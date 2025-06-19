import cv2
import numpy as np
import os
import pickle


video = cv2.VideoCapture(0)  # 0 for webcam
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()


facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if facedetect.empty():
    print("Error: Haar cascade file not found.")
    exit()


face_data = []  # Array to store face data
i = 0
name = input("Enter your Name: ").strip()


data_path = "data" # Create a directory for data if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

print(f"Collecting face data for {name}. Please stay in frame...")

# Start capturing video frames
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Save face data every 10 frames, up to 50 faces
        if len(face_data) < 50 and i % 10 == 0:
            face_data.append(resized_img)
            print(f"Captured face {len(face_data)}/50")

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(frame, f"Faces: {len(face_data)}/50",
                    org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(50, 50, 255), thickness=1)

    cv2.imshow("Collecting Face Data", frame)

    # Stop after collecting 50 images
    if len(face_data) >= 50:
        print("Collected 50 images. Saving data...")
        break

    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
        print("Exiting before completing data collection.")
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Reshape face data and save
face_data = np.array(face_data).reshape(50, -1)

# Save names
names_file = os.path.join(data_path, 'names.pkl')
if not os.path.exists(names_file):
    names = [name] * 50
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 50

with open(names_file, 'wb') as f:
    pickle.dump(names, f)

# Save face data
face_data_file = os.path.join(data_path, 'face_data.pkl')
if not os.path.exists(face_data_file):
    faces = face_data
else:
    with open(face_data_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)

with open(face_data_file, 'wb') as f:
    pickle.dump(faces, f)

print("✅ Face data and names saved successfully!")

