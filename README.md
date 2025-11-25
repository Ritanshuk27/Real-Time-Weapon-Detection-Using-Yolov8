# Real-Time Weapon Detection Using YOLOv8

This project is a real-time weapon detection system built using **YOLOv8**, **OpenCV**, and **PyTorch**.  
It can detect weapons (such as guns and knives) from live webcam feed or video files and raise an alert whenever a weapon is detected.

> 🔒 **Disclaimer:**  
> This project is strictly for **academic, research, and safety applications** only.  
> It must **not** be used for any illegal or harmful activities.

---

## 👨‍💻 Authors

- **Satyajeet Kumar**  
- **Ritanshu Kumar Singh**

---

## 📌 Features

- Real-time detection of weapons using **YOLOv8**.
- Supports **webcam**, **video files**, or **image** inputs (depending on how you run the script).
- Visual bounding boxes with class labels and confidence scores.
- **Alarm sound** (`alarm.mp3`) when a weapon is detected.
- Multiple ways to run:
  - GUI-based application (`guiapp.py`)  
  - Terminal-based detection (`weapon_detect_terminal.py`)
- Custom trained model weights (`best.pt`) based on YOLOv8.

---

## 🧱 Project Structure

> Note: This is a simplified view of important files in the repository.

```bash
Real-Time-Weapon-Detection-Using-Yolov8/
├── weapon_detection_server/      # (If used) Backend/server-related code
├── .gitignore
├── .python-version               # Python version used in the project
├── alarm.mp3                     # Alarm sound on weapon detection
├── best.pt                       # Trained YOLOv8 model weights
├── yolov8m.pt                    # Base YOLOv8 model (medium)
├── download_images.py            # Script (likely) used to download dataset images
├── train.py                      # Script (likely) used for model training
├── test.py                       # Script (likely) used for testing/evaluation
├── guiapp.py                     # GUI-based weapon detection app
├── weapon_detect_terminal.py     # Terminal-based weapon detection script
├── needs.txt                     # Python dependencies (requirements)
├── weapon_detection.png          # Demo/screenshot image
└── New Text Document.txt         # Misc notes (if any)
