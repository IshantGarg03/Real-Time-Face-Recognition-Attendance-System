import cv2
import os 
import numpy as np
import sqlite3
from datetime import datetime

# Database setup
def setup_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS persons
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 age INTEGER,
                 register_date TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 person_id INTEGER,
                 check_in_time TEXT,
                 FOREIGN KEY(person_id) REFERENCES persons(id))''')

    conn.commit()
    conn.close() 
    
# Insert a new person into the database
def insert_person(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    person_id = c.lastrowid
    conn.commit()
    conn.close() 
    return person_id


# Register new person
def register_person():
    name = input("Enter person's name: ")
    age = input("Enter person's age: ")
    register_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO persons (name, age, register_date) VALUES (?, ?, ?)",
              (name, age, register_date))
    person_id = c.lastrowid
    conn.commit()
    conn.close()

    os.makedirs(f"face_data/{person_id}", exist_ok=True)
    capture_face_samples(person_id, name)
    return person_id

# Capture face samples
def capture_face_samples(person_id, name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    sample_count = 1
    required_samples = 21

    print(f"Capturing face samples for {name}. Please look at the camera...") 

    while sample_count < required_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            filename = f"face_data/{person_id}/sample_{sample_count}.jpg"
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sample_count += 1

        cv2.putText(frame, f"Samples: {sample_count}/{required_samples}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Capturing Face Samples', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {sample_count} samples for {name}")

# Just logs time of presence, no recognition
def mark_attendance_dummy():
    print("Starting attendance logging (without recognition)... Press 'q' to stop.")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu 
def main():
    setup_database()

    while True:
        print("\nAttendance System Menu:")
        print("1. Register new person")
        print("2. Just start camera for face logging")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            register_person()
        elif choice == "2":
            mark_attendance_dummy()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 
