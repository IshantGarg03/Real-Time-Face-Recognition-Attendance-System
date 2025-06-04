# import cv2
# import os
# import numpy as np
# from datetime import datetime

# def main():
#     # Create directory for storing face data if it doesn't exist
#     data_dir = "face_data"
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     person_name = input("Enter person's name: ")
#     person_dir = os.path.join(data_dir, person_name)
#     if not os.path.exists(person_dir):
#         os.makedirs(person_dir)
    
#     sample_count = 0
#     max_samples = 50  # Number of face samples to capture
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Save face samples
#             if sample_count < max_samples:
#                 face_img = gray[y:y+h, x:x+w]
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#                 filename = f"{person_dir}/{timestamp}.png"
#                 cv2.imwrite(filename, face_img)
#                 sample_count += 1
#                 print(f"Saved sample {sample_count}/{max_samples}")
        
#         cv2.putText(frame, f"Samples: {sample_count}/{max_samples}", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.imshow('Face Data Collection', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= max_samples:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Face data collection complete for {person_name}")

# if __name__ == "__main__":
#     main()

# print(cv2.__version__)















# import cv2
# import os

# def main():
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
    
#     # Check if webcam opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         print("Trying alternate camera index...")
#         cap = cv2.VideoCapture(1)  # Try index 1 if 0 fails
#         if not cap.isOpened():
#             print("Failed to open any camera.")
#             return
    
#     # Create directory for captures
#     os.makedirs("captures", exist_ok=True)
#     capture_count = 0
    
#     print("Webcam opened successfully!")
#     print("Press 'c' to capture, 'q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break
        
#         # Display the frame
#         cv2.imshow('Webcam Feed', frame)
        
#         # Check for key presses
#         key = cv2.waitKey(1)
        
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF == ord('c'):
#             # Save capture
#             filename = f"captures/capture_{capture_count}.jpg"
#             cv2.imwrite(filename, frame)
#             print(f"Saved {filename}")
#             capture_count += 1
    
#     # Clean up
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam released.")

# if __name__ == "__main__":
#     main()













# import cv2
# import os
# import numpy as np
# import sqlite3
# from datetime import datetime
# import face_recognition
# import pickle
# import sys

# # Database setup
# def setup_database():
#     conn = sqlite3.connect('attendance.db')
#     c = conn.cursor()
    
#     # Create persons table if not exists
#     c.execute('''CREATE TABLE IF NOT EXISTS persons
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                  name TEXT NOT NULL,
#                  age INTEGER,
#                  register_date TEXT)''')
    
#     # Create attendance table if not exists
#     c.execute('''CREATE TABLE IF NOT EXISTS attendance
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                  person_id INTEGER,
#                  check_in_time TEXT,
#                  FOREIGN KEY(person_id) REFERENCES persons(id))''')
    
#     conn.commit()
#     conn.close()

# # Register new person
# def register_person():
#     name = input("Enter person's name: ")
#     age = input("Enter person's age: ")
#     register_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     # Save to database
#     conn = sqlite3.connect('attendance.db')
#     c = conn.cursor()
#     c.execute("INSERT INTO persons (name, age, register_date) VALUES (?, ?, ?)",
#               (name, age, register_date))
#     person_id = c.lastrowid
#     conn.commit()
#     conn.close()
    
#     # Create directory for face data
#     os.makedirs(f"face_data/{person_id}", exist_ok=True)
    
#     # Capture face samples
#     capture_face_samples(person_id, name)
    
#     return person_id

# # Capture multiple face samples
# def capture_face_samples(person_id, name):
#     cap = cv2.VideoCapture(0)
#     sample_count = 0
#     required_samples = 20  # Number of samples to capture
    
#     print(f"Capturing face samples for {name}. Please move your head slowly...")
    
#     while sample_count < required_samples:
#         ret, frame = cap.read()
#         if not ret:
#             continue
            
#         # Convert to RGB (face_recognition uses RGB)
#         rgb_frame = frame[:, :, ::-1]
        
#         # Find faces
#         face_locations = face_recognition.face_locations(rgb_frame)
        
#         for (top, right, bottom, left) in face_locations:
#             # Save face sample
#             face_image = rgb_frame[top:bottom, left:right]
#             filename = f"face_data/{person_id}/sample_{sample_count}.jpg"
#             cv2.imwrite(filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            
#             # Draw rectangle
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             sample_count += 1
            
#         # Display count
#         cv2.putText(frame, f"Samples: {sample_count}/{required_samples}", 
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.imshow('Capturing Face Samples', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Captured {sample_count} samples for {name}")
    
#     # Train the model after capturing
#     train_model()

# # Train face recognition model
# def train_model():
#     known_encodings = []
#     known_ids = []
    
#     # Load all saved faces
#     for person_id in os.listdir("face_data"):
#         person_dir = os.path.join("face_data", person_id)
        
#         if os.path.isdir(person_dir):
#             for image_file in os.listdir(person_dir):
#                 image_path = os.path.join(person_dir, image_file)
#                 image = face_recognition.load_image_file(image_path)
                
#                 # Get face encodings
#                 encodings = face_recognition.face_encodings(image)
#                 if encodings:
#                     known_encodings.append(encodings[0])
#                     known_ids.append(int(person_id))
    
#     # Save the model
#     data = {"encodings": known_encodings, "ids": known_ids}
#     with open("face_model.pkl", "wb") as f:
#         pickle.dump(data, f)
    
#     print("Model trained with", len(known_ids), "persons")

# # Recognize faces and mark attendance
# def recognize_faces():
#     # Load trained model
#     try:
#         with open("face_model.pkl", "rb") as f:
#             data = pickle.load(f)
#     except:
#         print("No trained model found. Please register people first.")
#         return
    
#     cap = cv2.VideoCapture(0)
#     processed_ids = []  # To avoid duplicate entries
    
#     print("Starting face recognition for attendance...")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
            
#         # Convert to RGB
#         rgb_frame = frame[:, :, ::-1]
        
#         # Find faces
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # Compare with known faces
#             matches = face_recognition.compare_faces(data["encodings"], face_encoding)
#             person_id = None
            
#             if True in matches:
#                 matched_idx = matches.index(True)
#                 person_id = data["ids"][matched_idx]
                
#                 # Mark attendance if not already processed
#                 if person_id not in processed_ids:
#                     mark_attendance(person_id)
#                     processed_ids.append(person_id)
                    
#                     # Get person info from database
#                     conn = sqlite3.connect('attendance.db')
#                     c = conn.cursor()
#                     c.execute("SELECT name FROM persons WHERE id=?", (person_id,))
#                     name = c.fetchone()[0]
#                     conn.close()
                    
#                     print(f"Attendance marked for {name}")
            
#             # Draw rectangle and label
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             label = f"ID: {person_id}" if person_id else "Unknown"
#             cv2.putText(frame, label, (left, top-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         cv2.imshow('Face Recognition Attendance', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()

# # Mark attendance in database
# def mark_attendance(person_id):
#     check_in_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     conn = sqlite3.connect('attendance.db')
#     c = conn.cursor()
#     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)",
#               (person_id, check_in_time))
#     conn.commit()
#     conn.close()

# # Main menu
# def main():
#     setup_database()
    
#     while True:
#         print("\nAttendance System Menu:")
#         print("1. Register new person")
#         print("2. Start attendance marking")
#         print("3. Exit")
        
#         choice = input("Enter your choice: ")
        
#         if choice == "1":
#             register_person()
#         elif choice == "2":
#             recognize_faces()
#         elif choice == "3":
#             break
#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()

























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