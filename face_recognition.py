# import cv2
# import os
# import numpy as np
# from datetime import datetime

# # Load the Haar cascade and recognizer
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # Map label to name
# def get_label_name_map(base_folder="captured_faces"):
#     label_map = {}
#     current_label = 0
#     for name in os.listdir(base_folder):
#         if os.path.isdir(os.path.join(base_folder, name)):
#             label_map[current_label] = name
#             current_label += 1
#     return label_map

# label_map = get_label_name_map()

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         label, confidence = recognizer.predict(roi_gray)

#         name = label_map.get(label, "Unknown")

#         # Draw rectangle and label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()













import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Load models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Load label map
label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_map[int(label_id)] = name

# Connect to DB
conn = sqlite3.connect("attendance.db")
c = conn.cursor()

# Session status dictionary: name -> last_seen_time
last_seen = {}

# Time after which a logout is logged (in seconds)
LOGOUT_THRESHOLD = 30

def get_person_id_by_name(name):
    c.execute("SELECT id FROM persons WHERE name=?", (name,))
    row = c.fetchone()
    return row[0] if row else None

def log_check_in(person_id):
    check_in_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, check_in_time))
    conn.commit()
    print(f"[IN] ✅ {person_id} at {check_in_time}")

def log_check_out(person_id):
    check_out_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Find the latest check-in without checkout
    c.execute("""
        SELECT id FROM attendance 
        WHERE person_id=? AND check_out_time IS NULL 
        ORDER BY id DESC LIMIT 1
    """, (person_id,))
    row = c.fetchone()
    if row:
        attendance_id = row[0] 
        c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (check_out_time, attendance_id))
        conn.commit()
        print(f"[OUT] ✅ {person_id} at {check_out_time}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    now = datetime.now()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        name = label_map.get(label, "Unknown")

        if name != "Unknown":
            person_id = get_person_id_by_name(name)

            # Check last seen time
            if name not in last_seen:
                # New session
                log_check_in(person_id)
            else:
                time_diff = (now - last_seen[name]).total_seconds()
                if time_diff > LOGOUT_THRESHOLD:
                    log_check_out(person_id)
                    log_check_in(person_id)  # New session starts again

            # Update last seen
            last_seen[name] = now

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Attendance with Login/Logout", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
