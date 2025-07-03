import logging
import os
from datetime import datetime, timedelta
import base64
import cv2
import numpy as np
import sqlite3
import threading
from collections import deque, Counter
from io import BytesIO
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import csv
from PIL import Image
from train_model import train_and_save_model

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logger.error("Failed to load haarcascade_frontalface_default.xml")
    raise RuntimeError("Failed to load face cascade classifier")

register_capture_count = 0
new_user_name = None
new_user_label_id = None
recent_register_predictions = deque(maxlen=5)

csv_file = "attendance_log.csv"
if not os.path.exists(csv_file):
    logger.info(f"Creating new CSV file: {csv_file}")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Action", "Timestamp", "Location"])

attendance_logged = False
last_gsheet_write = {}
processing_lock = threading.Lock()

def load_label_map():
    logger.debug("Loading label map from labels.txt")
    label_map = {}
    if os.path.exists("labels.txt"):
        try:
            with open("labels.txt", "r") as f:
                for line in f:
                    label_id, name = line.strip().split(",")
                    label_map[int(label_id)] = name
            logger.info(f"Loaded {len(label_map)} entries from labels.txt")
        except Exception as e:
            logger.error(f"Failed to load labels.txt: {e}", exc_info=True)
    else:
        logger.warning("labels.txt not found")
    return label_map

def update_label_map(label_id, name):
    logger.debug(f"Updating label map with label_id: {label_id}, name: {name}")
    try:
        with open("labels.txt", "a") as f:
            f.write(f"{label_id},{name}\n")
        logger.info(f"Added label_id: {label_id}, name: {name} to labels.txt")
    except Exception as e:
        logger.error(f"Failed to update labels.txt: {e}", exc_info=True)

def init_db():
    logger.info("Initializing attendance database")
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                check_in_time TEXT,
                check_out_time TEXT,
                check_in_location TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
    finally:
        conn.close()

if not os.path.exists("attendance.db"):
    logger.info("attendance.db not found, creating new database")
    init_db()

try:
    conn = sqlite3.connect("attendance.db", check_same_thread=False)
    c = conn.cursor()
    logger.info("Established database connection")
except sqlite3.Error as e:
    logger.error(f"Failed to connect to database: {e}", exc_info=True)
    raise

def insert_person(name):
    logger.debug(f"Inserting person: {name}")
    try:
        c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        conn.commit()
        person_id = c.lastrowid
        update_label_map(person_id, name)
        logger.info(f"Inserted person: {name}, ID: {person_id}")
        return person_id
    except sqlite3.Error as e:
        logger.error(f"Failed to insert person {name}: {e}", exc_info=True)
        raise

def write_to_gsheet(name, action, timestamp, location="Office"):
    global last_gsheet_write
    logger.debug(f"Attempting to write to Google Sheet: {name}, {action}, {timestamp}")
    try:
        now = datetime.now()
        last_write = last_gsheet_write.get(name)
        if last_write and (now - last_write) < timedelta(minutes=5):
            logger.info(f"Skipped Google Sheet write for {name}: Cooldown active")
            return

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name("Excel_Name.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Attendance").sheet1
        sheet.append_row([name, action, timestamp, location])
        last_gsheet_write[name] = now
        logger.info(f"Successfully wrote to Google Sheet: {name}, {action}, {timestamp}")
    except Exception as e:
        logger.error(f"Failed to write to Google Sheet: {e}", exc_info=True)

recognizer = cv2.face.LBPHFaceRecognizer_create() 
 
last_recognition = {}
label_map = load_label_map() 

if os.path.exists("trained_model.yml"):
    try:
        recognizer.read("trained_model.yml")
        logger.info("Loaded trained_model.yml") 
    except cv2.error as e:
        logger.error(f"Failed to load trained_model.yml: {e}", exc_info=True)
else:
    logger.warning("trained_model.yml not found")

FRAME_WINDOW = 5
REQUIRED_CONFIRMATIONS = 1
recent_predictions = deque(maxlen=FRAME_WINDOW)

def decode_base64_image(base64_string):
    logger.debug(f"Decoding base64 image (first 100 chars): {base64_string[:100]}...")
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        else:
            raise ValueError("Base64 string does not contain expected data URI prefix (data:image)")
        if not base64_string:
            raise ValueError("Base64 string is empty after removing prefix")
        img_data = base64.b64decode(base64_string, validate=True)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        logger.debug("Successfully decoded base64 image")
        return frame
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}", exc_info=True)
        raise ValueError(f"Invalid base64 image: {e}")

@app.route("/start_registration", methods=["POST"])
def start_registration():
    logger.info("Received /start_registration request")
    global new_user_name, new_user_label_id, register_capture_count, recent_register_predictions
    try:
        data = request.get_json()
        name = data.get("name", "").strip()
        logger.debug(f"Registration name: {name}")

        if not name:
            logger.warning("Invalid name provided")
            return jsonify({"status": "danger", "message": "Please enter a valid name."})

        new_user_label_id = insert_person(name)
        new_user_name = name
        register_capture_count = 1
        recent_register_predictions.clear()  # Clear any previous predictions
        logger.info(f"Started registration for {name}, label_id: {new_user_label_id}")
        return jsonify({"status": "success", "message": "Registration started."})
    except Exception as e:
        logger.error(f"Error in start_registration: {e}", exc_info=True)
        return jsonify({"status": "danger", "message": f"Error: {str(e)}"})

@app.route("/process_register_frame", methods=["POST"])
def process_register_frame():
    logger.info("Received /process_register_frame request")
    global register_capture_count, new_user_name, new_user_label_id, recent_register_predictions
    try:
        data = request.get_json()
        image_data = data.get("image")
        name = data.get("name")
        logger.debug(f"Processing frame for name: {name}")

        if not image_data or not name or not new_user_name or new_user_label_id is None:
            logger.warning("Invalid request or registration not started")
            return jsonify({"status": "danger", "message": "Invalid request or registration not started."})

        if name != new_user_name:
            logger.warning(f"Name mismatch: provided {name}, expected {new_user_name}")
            return jsonify({"status": "danger", "message": "Name mismatch. Please restart registration."})

        frame = decode_base64_image(image_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        logger.debug(f"Detected {len(faces)} faces")

        if len(faces) == 0:
            logger.info("No face detected")
            return jsonify({"status": "warning", "message": "No face detected. Please try again."})

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            logger.debug(f"Face image shape: {face_img.shape}, type: {face_img.dtype}")
            if face_img.size == 0:
                logger.warning("Empty face image detected, skipping prediction")
                continue

            # Check for duplicate face only if model exists
            if os.path.exists("trained_model.yml"):
                try:
                    label, confidence = recognizer.predict(face_img)
                    logger.debug(f"Predicted label: {label}, confidence: {confidence}")
                    # Lower confidence means better match; < 50 indicates likely duplicate
                    if confidence < 65:
                        recent_register_predictions.append(label)
                        # Require 3 consistent predictions to confirm duplicate
                        if recent_register_predictions.count(label) >= 1:
                            register_capture_count = 0
                            new_user_name = None
                            new_user_label_id = None
                            recent_register_predictions.clear()
                            logger.warning(f"Duplicate face detected: label {label}, name {label_map.get(label, 'Unknown')}")
                            return jsonify({
                                "status": "danger",
                                "message": f"Face already registered. Cannot register again.",
                                # "message": f"Face already registered as {label_map.get(label, 'Unknown')}. Cannot register again.",
                                "complete": True
                            })
                    else:
                        logger.debug(f"Confidence {confidence} >= 65, likely a new face")  
                except cv2.error as e:
                    logger.error(f"Face recognition error: {e}", exc_info=True)
            else:
                logger.info("No trained model found, skipping duplicate face check")

            # Capture face sample if not a duplicate
            if 0 < register_capture_count <= 5:
                save_dir = f"face_data/{new_user_label_id}_{new_user_name}"
                os.makedirs(save_dir, exist_ok=True)
                img_path = os.path.join(save_dir, f"sample_{register_capture_count}.jpg")
                cv2.imwrite(img_path, face_img)
                logger.info(f"Saved face sample {register_capture_count}/5 to {img_path}")
                register_capture_count += 1

                if register_capture_count > 5:
                    register_capture_count = 0
                    new_user_name = None
                    new_user_label_id = None
                    recent_register_predictions.clear()
                    logger.info("Starting model retraining")
                    threading.Thread(target=train_and_save_model).start()
                    return jsonify({
                        "status": "success",
                        "message": "Registration complete. Model retraining started.",
                        "complete": True
                    })

            return jsonify({
                "status": "info",
                "message": f"Capturing sample {register_capture_count}/5"
            })

    except Exception as e:
        logger.error(f"Error in process_register_frame: {e}", exc_info=True)
        return jsonify({"status": "danger", "message": f"Error: {str(e)}"})

@app.route("/process_attendance_frame", methods=["POST"])
def process_attendance_frame():
    logger.info("Received /process_attendance_frame request")
    global recent_predictions, last_recognition, attendance_logged
    try:
        with processing_lock:
            if attendance_logged:
                recent_predictions.clear()
                logger.info("Attendance already logged for this session")
                return jsonify({
                    "status": "info",
                    "message": "Attendance already logged for this session.",
                    "complete": True,
                    "start_webcam": True 
                }) 

            data = request.get_json()
            image_data = data.get("image")
            logger.debug("Received attendance frame data")

            if not image_data:
                logger.warning("No image data provided")
                return jsonify({
                    "status": "error",
                    "message": "No image data provided.",
                    "complete": True,
                    "stop_webcam": True
                })

            frame = decode_base64_image(image_data)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Fixed: BGR2BGR -> BGR2GRAY
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            name = "Unknown"
            logger.debug(f"Detected {len(faces)} faces")

            if len(faces) == 0:
                logger.info("No face detected")
                return jsonify({
                    "status": "info",
                    "message": "No face detected.",
                    "complete": False,
                    "stop_webcam": False
                })

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                if not os.path.exists("trained_model.yml"):
                    logger.info("No trained model found, cannot process attendance")
                    return jsonify({
                        "status": "warning",
                        "message": "No trained model available. Please register users first.",
                        "complete": True,
                        "stop_webcam": True
                    })

                try:
                    label, confidence = recognizer.predict(face_roi)
                    logger.debug(f"Predicted label: {label}, confidence: {confidence}")
                    if confidence < 65:
                        name = label_map.get(label, "Unknown")
                    else:
                        name = "Unknown"
                except cv2.error as e:
                    logger.error(f"Face recognition error: {e}", exc_info=True)
                    return jsonify({
                        "status": "error",
                        "message": f"Recognition error: {str(e)}",
                        "complete": True,
                        "stop_webcam": True
                    })

                now = datetime.now()  
                last_time = last_recognition.get(name)
                if name != "Unknown" and last_time and (now - last_time) < timedelta(minutes=30):
                    recent_predictions.clear() 
                    attendance_logged = True
                    logger.info(f"Cooldown active for {name}")
                    return jsonify({
                        "status": "info",
                        "message": f"Attendance already logged for {name} recently.",
                        "complete": True,
                        "stop_webcam": True
                    })

                recent_predictions.append(name)

                if len(recent_predictions) >= REQUIRED_CONFIRMATIONS:
                    most_common = Counter(recent_predictions).most_common(1)[0]
                    if most_common[0] != "Unknown" and most_common[1] >= REQUIRED_CONFIRMATIONS:
                        recognized_name = most_common[0]
                        logger.debug(f"Recognized name: {recognized_name}")

                        c.execute("SELECT id FROM persons WHERE name=?", (recognized_name,))
                        row = c.fetchone()

                        if row:
                            person_id = row[0]
                            timestamp = now.strftime('%d-%m-%Y %H:%M:%S') 
                            action = "IN"

                            c.execute("""SELECT check_in_time, check_out_time FROM attendance 
                                         WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
                            record = c.fetchone()

                            if not record or record[1] is not None:
                                c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)",
                                          (person_id, timestamp))
                                action = "IN"
                            else:
                                c.execute("UPDATE attendance SET check_out_time=? WHERE person_id=? AND check_out_time IS NULL",
                                          (timestamp, person_id))
                                action = "OUT"

                            conn.commit()
                            logger.info(f"Database updated: {action} for {recognized_name} at {timestamp}")

                            with open(csv_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([recognized_name, action, timestamp, "Office"])
                                logger.info(f"Logged to CSV: {recognized_name}, {action}, {timestamp}")

                            write_to_gsheet(recognized_name, action, timestamp, "Office")

                            last_recognition[recognized_name] = now
                            recent_predictions.clear()
                            attendance_logged = True
                            logger.info(f"Attendance logged: {action} for {recognized_name} at {timestamp}")
                            return jsonify({
                                "status": "success",
                                "message": f"Attendance logged: {action} for {recognized_name} at {timestamp}",
                                "complete": True,
                                "stop_webcam": True
                            })
                break

            return jsonify({
                "status": "info",
                "message": f"Detected: {name}",
                "complete": False,
                "stop_webcam": False
            })
    except Exception as e:
        logger.error(f"Error in process_attendance_frame: {e}", exc_info=True)
        attendance_logged = False  # Reset on error to allow retries
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}",
            "complete": True,
            "stop_webcam": True
        })

@app.route("/")
def index():
    logger.info("Accessed / route")
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    logger.info("Accessed /register route")
    try:
        if request.method == "POST":
            logger.debug("Received POST request for /register")
            flash("Please use the form on the registration page.", "info")
            return redirect(url_for("register"))
        return render_template("register.html")
    except Exception as e:
        logger.error(f"Error in register route: {e}", exc_info=True)
        return render_template("error.html"), 500

@app.route("/attendance")
def attendance():
    logger.info("Accessed /attendance route")
    global attendance_logged
    try:
        attendance_logged = False
        logger.info("Reset attendance_logged flag to False")
        return render_template("attendance.html")
    except Exception as e:
        logger.error(f"Error in attendance route: {e}", exc_info=True)
        return render_template("error.html"), 500

@app.route("/view_attendance", methods=["GET"])
def view_attendance():
    logger.info("Accessed /view_attendance route")
    try:
        c.execute("SELECT id, name FROM persons")
        persons = [{"id": row[0], "name": row[1]} for row in c.fetchall()]
        selected_person_id = request.args.get("person_id", type=int)
        attendance_records = []
        logger.debug(f"Selected person_id: {selected_person_id}")

        if selected_person_id:
            c.execute("""
                SELECT check_in_time, check_out_time, check_in_location
                FROM attendance 
                WHERE person_id = ? 
                ORDER BY check_in_time DESC
            """, (selected_person_id,))
            attendance_records = [
                {"check_in_time": row[0], "check_out_time": row[1], "check_in_location": row[2]}
                for row in c.fetchall()
            ]
            logger.info(f"Retrieved {len(attendance_records)} attendance records for person_id: {selected_person_id}")

        return render_template(
            "view_attendance.html",
            persons=persons,
            selected_person_id=selected_person_id,
            attendance_records=attendance_records
        )
    except sqlite3.Error as e:
        logger.error(f"Database error in view_attendance: {e}", exc_info=True)
        return render_template("error.html"), 500
    except Exception as e:
        logger.error(f"Error in view_attendance: {e}", exc_info=True)
        return render_template("error.html"), 500

@app.route("/help")
def help_page():
    logger.info("Accessed /help route")
    try:
        return render_template("help.html")
    except Exception as e:
        logger.error(f"Error in help route: {e}", exc_info=True)
        return render_template("error.html"), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    try:
        app.run(debug=True, host="0.0.0.0")
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
    finally:
        logger.info("Closing database connection")
        conn.close() 
