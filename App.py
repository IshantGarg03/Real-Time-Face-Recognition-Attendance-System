# # import streamlit as st
# # import pandas as pd
# # from datetime import datetime

# # st.set_page_config(page_title="Attendance Dashboard", layout="wide")

# # st.title("📊 Real-Time Attendance Dashboard")

# # # ✅ Load data from CSV with safety check
# # @st.cache_data
# # def load_data():
# #     try:
# #         df = pd.read_csv("attendance_log.csv", engine="python", on_bad_lines="skip")
# #         return df
# #     except FileNotFoundError:
# #         st.warning("⚠️ No attendance_log.csv file found.")
# #         return pd.DataFrame(columns=["Name", "Action", "Timestamp", "Location"])

# # df = load_data()

# # # ✅ Check if there's data to display
# # if df.empty:
# #     st.info("No attendance data available yet.")
# #     st.stop()

# # # ✅ Convert Timestamp to datetime for filtering
# # df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
# # df.dropna(subset=['Timestamp'], inplace=True)  # Drop bad rows
# # df['Date'] = df['Timestamp'].dt.date

# # # --- Filters ---
# # st.sidebar.header("📌 Filter Records")

# # # Filter by Name
# # names = df['Name'].dropna().unique().tolist()
# # selected_name = st.sidebar.selectbox("Select Name", ["All"] + names)

# # # Filter by Date 
# # selected_date = st.sidebar.date_input("Select Date (optional)")

# # # --- Apply Filters ---
# # if selected_name != "All":
# #     df = df[df['Name'] == selected_name]

# # if selected_date:
# #     df = df[df['Date'] == selected_date]

# # # ✅ Display Filtered Records
# # st.markdown(f"### Showing {len(df)} records")
# # st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

# # # Optional: Show raw CSV file download
# # st.download_button("⬇ Download Full CSV", data=df.to_csv(index=False), file_name="filtered_attendance.csv") 






# from flask import Flask, render_template, Response, request, redirect, url_for, flash
# import cv2
# import os
# import numpy as np
# from datetime import datetime
# import sqlite3
# import threading
# from collections import deque, Counter

# # Import your training function (make sure you have train_and_save_model in train_model.py)
# from train_model import train_and_save_model

# app = Flask(__name__)
# app.secret_key = "your_secret_key_here"  # For flashing messages

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Global variables for registration
# camera = cv2.VideoCapture(0)
# register_capture_count = 0
# new_user_name = None
# new_user_label_id = None

# # Load label map for attendance recognition
# def load_label_map():
#     label_map = {}
#     if os.path.exists("labels.txt"):
#         with open("labels.txt", "r") as f:
#             for line in f:
#                 label_id, name = line.strip().split(",")
#                 label_map[int(label_id)] = name
#     return label_map

# label_map = load_label_map()

# # SQLite connection (allow multiple threads)
# conn = sqlite3.connect("attendance.db", check_same_thread=False)
# c = conn.cursor()

# # Helper: insert new person to DB and return id
# def insert_person(name):
#     c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
#     conn.commit()
#     return c.lastrowid

# # --- Registration Video Stream Generator ---
# def gen_register():
#     global register_capture_count, new_user_label_id, new_user_name

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             if register_capture_count > 0 and register_capture_count <= 20:
#                 # Save face ROI image for training
#                 face_img = gray[y:y+h, x:x+w]
#                 save_dir = f"face_data/{new_user_label_id}_{new_user_name}"
#                 os.makedirs(save_dir, exist_ok=True)
#                 img_path = os.path.join(save_dir, f"sample_{register_capture_count}.jpg")
#                 cv2.imwrite(img_path, face_img)

#                 register_capture_count += 1

#                 if register_capture_count > 20:
#                     register_capture_count = 0
#                     # Retrain model in background thread after capture completes
#                     threading.Thread(target=train_and_save_model).start()
#                     flash("Registration complete and model retraining started.", "success")

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# # --- Attendance Recognition Setup ---
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# if os.path.exists("trained_model.yml"):
#     recognizer.read("trained_model.yml")
# else:
#     print("[Warning] trained_model.yml not found! Please train the model first.")

# FRAME_WINDOW = 5
# REQUIRED_CONFIRMATIONS = 3
# recent_predictions = deque(maxlen=FRAME_WINDOW)

# attendance_camera = cv2.VideoCapture(0)

# def gen_attendance():
#     global recent_predictions
#     while True:
#         ret, frame = attendance_camera.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         name = "Unknown"

#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             label, confidence = recognizer.predict(face_roi)
#             if confidence < 65:
#                 name = label_map.get(label, "Unknown")
#             else:
#                 name = "Unknown"

#             recent_predictions.append(name)

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         # Confirm consistent predictions before logging attendance
#         if len(recent_predictions) == FRAME_WINDOW:
#             most_common = Counter(recent_predictions).most_common(1)[0]
#             if most_common[0] != "Unknown" and most_common[1] >= REQUIRED_CONFIRMATIONS:
#                 recognized_name = most_common[0]

#                 c.execute("SELECT id FROM persons WHERE name=?", (recognized_name,))
#                 row = c.fetchone()

#                 if row:
#                     person_id = row[0]

#                     # Check last attendance record
#                     c.execute("""SELECT check_in_time, check_out_time FROM attendance
#                                  WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#                     record = c.fetchone()

#                     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#                     if not record or record[1] is not None:
#                         # Log check-in
#                         c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)",
#                                   (person_id, now))
#                         conn.commit()
#                         print(f"[IN] {recognized_name} at {now}")
#                     else:
#                         # Log check-out
#                         c.execute("UPDATE attendance SET check_out_time=? WHERE person_id=? AND check_out_time IS NULL",
#                                   (now, person_id))
#                         conn.commit()
#                         print(f"[OUT] {recognized_name} at {now}")

#                     recent_predictions.clear()  # reset after logging

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# # --- Flask Routes ---

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/register", methods=["GET", "POST"])
# def register():
#     global new_user_name, new_user_label_id, register_capture_count
#     if request.method == "POST":
#         name = request.form.get("name").strip()
#         if not name:
#             flash("Please enter a valid name.", "danger")
#             return redirect(url_for("register"))

#         # Check if user already registered
#         c.execute("SELECT id FROM persons WHERE name=?", (name,))
#         row = c.fetchone()
#         if row:
#             flash("User already registered.", "warning")
#             return redirect(url_for("register"))

#         # Insert person and start capturing images
#         person_id = insert_person(name)
#         new_user_label_id = person_id
#         new_user_name = name
#         register_capture_count = 1  # start capturing from next frame

#         flash("Name registered. Please look at the camera for face capture.", "info")
#         return redirect(url_for("register"))

#     return render_template("register.html")

# @app.route("/video_feed_register")
# def video_feed_register():
#     return Response(gen_register(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/attendance")
# def attendance():
#     return render_template("attendance.html")

# @app.route("/video_feed_attendance")
# def video_feed_attendance():
#     return Response(gen_attendance(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     try:
#         app.run(debug=True)
#     finally:
#         camera.release()
#         attendance_camera.release()
#         conn.close()





















from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import os
import sys
import signal
import numpy as np
from datetime import datetime
import sqlite3
import threading
from collections import deque, Counter
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import csv

# Import training function
from train_model import train_and_save_model

app = Flask(__name__)
app.secret_key = "40b25abfe26e57b7b0d472caed228d37"

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Global variables
register_capture_count = 0
new_user_name = None
new_user_label_id = None

# CSV logging
csv_file = "attendance_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Action", "Timestamp", "Location"])

# Label map loader
def load_label_map():
    label_map = {}
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            for line in f:
                label_id, name = line.strip().split(",")
                label_map[int(label_id)] = name
    return label_map

# SQLite setup
conn = sqlite3.connect("attendance.db", check_same_thread=False)
c = conn.cursor()

def insert_person(name):
    c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    conn.commit()
    return c.lastrowid

# Google Sheet logging
def write_to_gsheet(name, action, timestamp, location="Office"):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name("Attendance.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Attendance").sheet1
        sheet.append_row([name, action, timestamp, location]) 
    except Exception as e: 
        print("Failed to write to Google Sheet:", e)

# Registration Video Stream
def gen_register():
    global register_capture_count, new_user_label_id, new_user_name
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("❌ Error: Cannot access webcam for registration.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to capture frame during registration.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if 0 < register_capture_count <= 20:
                face_img = gray[y:y + h, x:x + w]
                save_dir = f"face_data/{new_user_label_id}_{new_user_name}"
                os.makedirs(save_dir, exist_ok=True)
                img_path = os.path.join(save_dir, f"sample_{register_capture_count}.jpg")
                cv2.imwrite(img_path, face_img)

                register_capture_count += 1

                if register_capture_count > 20:
                    register_capture_count = 0
                    threading.Thread(target=train_and_save_model).start()
                    flash("Registration complete and model retraining started.", "success")

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    camera.release()

# Attendance Recognition Stream
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = {}

if os.path.exists("trained_model.yml"):
    recognizer.read("trained_model.yml")
    label_map = load_label_map()
else:
    print("[Warning] trained_model.yml not found!")

FRAME_WINDOW = 5
REQUIRED_CONFIRMATIONS = 3
recent_predictions = deque(maxlen=FRAME_WINDOW)

def gen_attendance():
    global recent_predictions
    attendance_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not attendance_camera.isOpened():
        print("❌ Error: Cannot access webcam for attendance.")
        return

    while True:
        ret, frame = attendance_camera.read()
        if not ret:
            print("❌ Failed to capture frame during attendance.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        name = "Unknown"

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_roi)

            if confidence < 65:
                name = label_map.get(label, "Unknown")
            else:
                name = "Unknown"

            recent_predictions.append(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if len(recent_predictions) == FRAME_WINDOW:
            most_common = Counter(recent_predictions).most_common(1)[0]
            if most_common[0] != "Unknown" and most_common[1] >= REQUIRED_CONFIRMATIONS:
                recognized_name = most_common[0]
                c.execute("SELECT id FROM persons WHERE name=?", (recognized_name,))
                row = c.fetchone()

                if row:
                    person_id = row[0]
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    action = "IN"

                    c.execute("""SELECT check_in_time, check_out_time FROM attendance
                                 WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
                    record = c.fetchone()

                    if not record or record[1] is not None:
                        c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)",
                                  (person_id, now))
                        action = "IN"
                    else:
                        c.execute("UPDATE attendance SET check_out_time=? WHERE person_id=? AND check_out_time IS NULL",
                                  (now, person_id))
                        action = "OUT"

                    conn.commit()

                    with open("attendance_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([recognized_name, action, now, "Office"])

                    write_to_gsheet(recognized_name, action, now, "Office")
                    recent_predictions.clear()
                    print(f"[{action}] {recognized_name} at {now}")
                    print("Attendance logged. Shutting down the server...")
                    os.kill(os.getpid(), signal.SIGINT)

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    attendance_camera.release()

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    global new_user_name, new_user_label_id, register_capture_count
    if request.method == "POST":
        name = request.form.get("name").strip()
        if not name:
            flash("Please enter a valid name.", "danger")
            return redirect(url_for("register"))

        c.execute("SELECT id FROM persons WHERE name=?", (name,))
        if c.fetchone():
            flash("User already registered.", "warning")
            return redirect(url_for("register"))

        person_id = insert_person(name)
        new_user_label_id = person_id
        new_user_name = name
        register_capture_count = 1
        flash("Name registered. Please look at the camera for face capture.", "info")
        return redirect(url_for("register"))

    return render_template("register.html")

@app.route("/video_feed_register")
def video_feed_register():
    return Response(gen_register(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/video_feed_attendance")
def video_feed_attendance():
    return Response(gen_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        conn.close() 
