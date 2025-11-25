import cv2
from ultralytics import YOLO
import pygame
import time

# Initialize YOLO
model = YOLO("yolov8m.pt")

# Initialize pygame for alarm
pygame.mixer.init()
alarm_sound = "alarm.mp3"

# Open webcam
cap = cv2.VideoCapture(0)

# Weapon classes
weapon_classes = ["knife", "gun"]

alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # YOLO inference
    detected = False

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        class_name = model.names[int(cls)]
        if class_name.lower() in weapon_classes:
            detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if detected and not alarm_playing:
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play(-1)  # loop until stopped
        alarm_playing = True
    elif not detected and alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

    cv2.imshow("Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
