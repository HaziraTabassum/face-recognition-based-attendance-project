# 🎯 Face Recognition Based Attendance System

A Python-based application that uses face recognition to automate attendance marking.

---

## 📌 Features

- ✨ Real-time face detection and recognition using OpenCV
- ✅ Attendance stored in a CSV file
- 💾 Trains and stores facial data using Pickle
- 🔍 Uses Haar Cascade classifier for face detection
- 📈 Lightweight and fast implementation using KNN

---

## 🧠 Technologies Used

- Python 3.13
- OpenCV
- NumPy
- Pickle
- Haarcascade frontal face detection

---

## 🛠️ Project Structure

```
├── .venv/
│   ├── Lib/
│   ├── Scripts/
│   ├── .gitignore
│   └── pyvenv.cfg
├── Background/
│   └── img.png
├── data/
│   ├── face_data.pkl
│   ├── names.pkl
│   └── attendance.csv
├── attendance.py
├── Dataset.py
├── haarcascade_frontalface_default.xml
└── README.md
```

---

## 🚀 How to Run

1. Clone this repo:
🔗 https://github.com/HaziraTabassum/face-recognition-based-attendance-project

2. Install dependencies:
pip install -r requirements.txt

3. Run to collect dataset:
python Dataset.py
4. Run to recognize and mark attendance:
python attendance.py


##  Acknowledgements

- Haarcascade from OpenCV GitHub
- Inspired by AI attendance systems

---
## ✍️ Author

**Hazira Tabbassum**  
📧 [tabassumhazira48@gmail.com](mailto:tabassumhazira48@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/hazira-tabassum/)

---

⭐ Don’t forget to give this project a star if you find it useful!


