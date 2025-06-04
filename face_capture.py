import cv2
import os
from datetime import datetime
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

person_name = input("Enter person's name: ")

cap = cv2.VideoCapture(0)

image_count = -1 
total_images = 30  

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]

        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_stamp = now.strftime("%H-%M-%S-%f")  

        folder = f"captured_faces/{person_name}/{date_folder}"
        os.makedirs(folder, exist_ok=True)

        cv2.imwrite(f"{folder}/face_{time_stamp}.jpg", face)
        image_count += 1
        print(f"[INFO] Captured image {image_count}/{total_images}")

        time.sleep(0.2)  

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= total_images:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Face capture completed.") 