# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION ONLY IN SESSION


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # --- Google Sheet Logging (optional) ---
# def log_to_gsheet(name, action):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp])
#         print(f"[Google Sheet] {action} logged for {name}")
#     except Exception as e:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp])
#     print(f"[CSV] {action} logged for {name}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now))
#     conn.commit()
#     log_to_csv(name, "IN")
#     log_to_gsheet(name, "IN")
#     print(f"[IN] {name} at {now}")

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (now, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT")
#         log_to_gsheet(name, "OUT")
#         print(f"[OUT] {name} at {now}")

# # --- Main ---
# cap = cv2.VideoCapture(0)
# last_seen = {}
# LOGOUT_THRESHOLD = 30  # seconds

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     now = datetime.now()

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         label, confidence = recognizer.predict(roi_gray)
#         name = label_map.get(label, "Unknown")

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             if name not in last_seen:
#                 log_check_in(person_id, name)
#             else:
#                 time_diff = (now - last_seen[name]).total_seconds()
#                 if time_diff > LOGOUT_THRESHOLD:
#                     log_check_out(person_id, name)
#                     log_check_in(person_id, name)  # Restart new session

#             last_seen[name] = now

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# conn.close()
# cv2.destroyAllWindows()































# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION CONTINUOUSLY IN AND OUT.


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # --- Google Sheet Logging (optional) ---
# def log_to_gsheet(name, action):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp])
#         print(f"[Google Sheet] {action} logged for {name}")
#     except Exception as e:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp])
#     print(f"[CSV] {action} logged for {name}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now))
#     conn.commit()
#     log_to_csv(name, "IN")
#     log_to_gsheet(name, "IN")
#     print(f"[IN] {name} at {now}")

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (now, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT")
#         log_to_gsheet(name, "OUT")
#         print(f"[OUT] {name} at {now}")

# # --- Main ---
# cap = cv2.VideoCapture(0)
# last_seen = {}        # Tracks last seen time
# last_logged = {}      # Tracks if currently IN
# LOGOUT_THRESHOLD = 30 # seconds

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     now = datetime.now()
#     current_names = set()

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         label, confidence = recognizer.predict(roi_gray)
#         name = label_map.get(label, "Unknown")

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             current_names.add(name)

#             if name not in last_logged:
#                 log_check_in(person_id, name)
#                 last_logged[name] = True

#             last_seen[name] = now

#             # Draw face box
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#             cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     # Check for OUT
#     for name in list(last_logged.keys()):
#         if name not in current_names:
#             time_diff = (now - last_seen.get(name, now)).total_seconds()
#             if time_diff > LOGOUT_THRESHOLD:
#                 person_id = get_person_id(name)
#                 if person_id:
#                     log_check_out(person_id, name)
#                     del last_logged[name]
#                     del last_seen[name]

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# conn.close()
# cv2.destroyAllWindows()





























# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION IN AND OUT BUT NOT CLOSE THE TERMINAL OR TERMINATE THE PROGRAM.


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # --- Google Sheet Logging ---
# def log_to_gsheet(name, action):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp])
#         print(f"[Google Sheet] {action} logged for {name}")
#     except Exception:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp])
#     print(f"[CSV] {action} logged for {name}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# # --- Main ---
# cap = cv2.VideoCapture(0)
# cooldown = {}
# COOLDOWN_SECONDS = 30  # Cooldown time in seconds

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     now_time = datetime.now()

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         label, confidence = recognizer.predict(roi_gray)
#         name = label_map.get(label, "Unknown")

#         if name != "Unknown":
#             if name in cooldown:
#                 time_diff = (now_time - cooldown[name]).total_seconds()
#                 if time_diff < COOLDOWN_SECONDS:
#                     continue  # Skip if still in cooldown

#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             # Fetch last attendance entry
#             c.execute("""SELECT id, check_in_time, check_out_time FROM attendance 
#                          WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#             last_entry = c.fetchone()
#             now_str = now_time.strftime('%Y-%m-%d %H:%M:%S')

#             if last_entry:
#                 att_id, check_in, check_out = last_entry
#                 if check_out is None:
#                     # Still IN, so log OUT
#                     c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (now_str, att_id))
#                     conn.commit()
#                     log_to_csv(name, "OUT")
#                     log_to_gsheet(name, "OUT")
#                     print(f"[OUT] {name} at {now_str}")
#                 else:
#                     # Last session completed, log new IN
#                     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now_str))
#                     conn.commit()
#                     log_to_csv(name, "IN")
#                     log_to_gsheet(name, "IN")
#                     print(f"[IN] {name} at {now_str}")
#             else:
#                 # No previous entry, log IN
#                 c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now_str))
#                 conn.commit()
#                 log_to_csv(name, "IN")
#                 log_to_gsheet(name, "IN")
#                 print(f"[IN] {name} at {now_str}")

#             cooldown[name] = now_time  # Update cooldown time

#         # Draw face box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# conn.close()
# cv2.destroyAllWindows()
































# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION IN AND OUT AND CLOSE THE TERMINAL BUT NOT ADD LOCATION.


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# import sys
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # --- Google Sheet Logging (optional) ---
# def log_to_gsheet(name, action):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp])
#         print(f"[Google Sheet] {action} logged for {name}")
#     except Exception as e:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp])
#     print(f"[CSV] {action} logged for {name}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now))
#     conn.commit()
#     log_to_csv(name, "IN")
#     log_to_gsheet(name, "IN")
#     print(f"[IN] {name} at {now}")
#     exit_program()

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (now, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT")
#         log_to_gsheet(name, "OUT")
#         print(f"[OUT] {name} at {now}")
#         exit_program()

# def exit_program():
#     cap.release()
#     conn.close()
#     cv2.destroyAllWindows()
#     sys.exit(0)  # Terminates the program immediately

# # --- Main ---
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

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             # Check if already IN, then log OUT. Else log IN.
#             # Check last record for this person
#             c.execute("""SELECT check_in_time, check_out_time FROM attendance 
#                          WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#             record = c.fetchone()
            
#             if not record:
#                 # No previous record at all — first time entry
#                 log_check_in(person_id, name)
#             elif record[1] is None:
#                 # Last record exists but has no check_out_time
#                 log_check_out(person_id, name)
#             else:
#                 # Last record exists and has both in & out — create new IN
#                 log_check_in(person_id, name)


#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# exit_program()

























# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION IN AND OUT AND CLOSE THE TERMINAL AND ADD LOCATION (ONLY CITY).


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# import sys
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# import geocoder  # <-- NEW import

# # --- Location fetching function ---
# def get_location():
#     try:
#         g = geocoder.ip('me')
#         city = g.city if g.city else "Unknown"
#         state = g.state if g.state else "Unknown"
#         return f"{city}, {state}"
#     except:
#         return "Unknown"

# # --- Google Sheet Logging (optional) ---
# def log_to_gsheet(name, action, location):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp, location])
#         print(f"[Google Sheet] {action} logged for {name} at {location}")
#     except Exception as e:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action, location):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp', 'Location'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp, location])
#     print(f"[CSV] {action} logged for {name} at {location}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_location()  # Get location here
#     c.execute("INSERT INTO attendance (person_id, check_in_time) VALUES (?, ?)", (person_id, now))
#     conn.commit()
#     log_to_csv(name, "IN", location)
#     log_to_gsheet(name, "IN", location)
#     print(f"[IN] {name} at {now} from {location}")
#     exit_program()

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_location()  # Get location here
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=? WHERE id=?", (now, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT", location)
#         log_to_gsheet(name, "OUT", location)
#         print(f"[OUT] {name} at {now} from {location}")
#         exit_program()

# def exit_program():
#     cap.release()
#     conn.close()
#     cv2.destroyAllWindows()
#     sys.exit(0)  # Terminates the program immediately

# # --- Main ---
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

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             # Check if already IN, then log OUT. Else log IN.
#             c.execute("""SELECT check_in_time, check_out_time FROM attendance 
#                          WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#             record = c.fetchone()
            
#             if not record:
#                 log_check_in(person_id, name)
#             elif record[1] is None:
#                 log_check_out(person_id, name)
#             else:
#                 log_check_in(person_id, name)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# exit_program()



























# THIS CODE HELPS IN TRACKING ATTENDANCE USING FACE RECOGNITION IN AND OUT AND CLOSE THE TERMINAL AND ADD LOCATION (INCORRECT DETAILED LOCATION).


# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# import sys
# from datetime import datetime
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# import geocoder
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut

# # --- Get detailed location using reverse geocoding ---
# def get_detailed_location():
#     try:
#         g = geocoder.ip('me')
#         latlng = g.latlng
#         if latlng is None:
#             return "Location Unknown"

#         geolocator = Nominatim(user_agent="attendance_app")
#         location = geolocator.reverse(f"{latlng[0]}, {latlng[1]}", language='en', exactly_one=True, timeout=10)
#         if location is None:
#             return "Location Unknown"

#         address = location.raw.get('address', {})
#         parts = []
#         for key in ['road', 'suburb', 'city_district', 'city', 'state', 'postcode', 'country']:
#             if key in address:
#                 parts.append(address[key])

#         detailed_address = ', '.join(parts)
#         return detailed_address if detailed_address else "Location Unknown"

#     except GeocoderTimedOut:
#         return "Location Timeout"
#     except Exception as e:
#         return "Location Error"

# # --- Google Sheet Logging ---
# def log_to_gsheet(name, action, location):
#     try:
#         scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#         creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
#         client = gspread.authorize(creds)
#         sheet = client.open("Attendance").sheet1
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         sheet.append_row([name, action, timestamp, location])
#         print(f"[Google Sheet] {action} logged for {name} at {location}")
#     except Exception as e:
#         import traceback
#         print("[Google Sheet Error]")
#         traceback.print_exc()

# # --- CSV Logging ---
# def log_to_csv(name, action, location):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp', 'Location'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp, location])
#     print(f"[CSV] {action} logged for {name} at {location}")

# # --- Load Recognizer and Cascade ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to SQLite ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_detailed_location()
#     c.execute("INSERT INTO attendance (person_id, check_in_time, check_in_location) VALUES (?, ?, ?)", (person_id, now, location))
#     conn.commit()
#     log_to_csv(name, "IN", location)
#     log_to_gsheet(name, "IN", location)
#     print(f"[IN] {name} at {now} - Location: {location}")
#     exit_program()

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_detailed_location()
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=?, check_out_location=? WHERE id=?", (now, location, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT", location)
#         log_to_gsheet(name, "OUT", location)
#         print(f"[OUT] {name} at {now} - Location: {location}")
#         exit_program()

# def exit_program():
#     cap.release()
#     conn.close()
#     cv2.destroyAllWindows()
#     sys.exit(0)

# # --- Main ---
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

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             c.execute("""SELECT check_in_time, check_out_time FROM attendance 
#                          WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#             record = c.fetchone()
            
#             if not record:
#                 log_check_in(person_id, name)
#             elif record[1] is None:
#                 log_check_out(person_id, name)
#             else:
#                 log_check_in(person_id, name)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     cv2.imshow("Attendance", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# exit_program() 































# import cv2
# import numpy as np
# import sqlite3
# import os
# import csv
# import sys
# from datetime import datetime
# import geocoder

# # --- Configuration ---
# CONFIDENCE_THRESHOLD = 60

# # --- Location Fetching ---
# def get_location():
#     try:
#         g = geocoder.ip('me')
#         city = g.city if g.city else "Unknown"
#         state = g.state if g.state else "Unknown"
#         return f"{city}, {state}"
#     except:
#         return "Unknown"

# # --- CSV Logging ---
# def log_to_csv(name, action, location):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     file_path = 'attendance_log.csv'
#     if not os.path.exists(file_path):
#         with open(file_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Name', 'Action', 'Timestamp', 'Location'])
#     with open(file_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, action, timestamp, location])
#     print(f"[CSV] {action} logged for {name} at {location}")

# # --- Exit Procedure ---
# def exit_program():
#     cap.release()
#     conn.close()
#     cv2.destroyAllWindows()
#     sys.exit(0)

# # --- Load Models ---
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trained_model.yml')

# # --- Load Label Map ---
# label_map = {}
# with open("labels.txt", "r") as f:
#     for line in f:
#         label_id, name = line.strip().split(",")
#         label_map[int(label_id)] = name

# # --- Connect to Database ---
# conn = sqlite3.connect("attendance.db")
# c = conn.cursor()

# def get_person_id(name):
#     c.execute("SELECT id FROM persons WHERE name=?", (name,))
#     row = c.fetchone()
#     return row[0] if row else None

# def log_check_in(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_location()
#     c.execute("INSERT INTO attendance (person_id, check_in_time, check_in_location) VALUES (?, ?, ?)", (person_id, now, location))
#     conn.commit()
#     log_to_csv(name, "IN", location)
#     print(f"[IN] {name} at {now} from {location}")
#     exit_program()

# def log_check_out(person_id, name):
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     location = get_location()
#     c.execute("""SELECT id FROM attendance 
#                  WHERE person_id=? AND check_out_time IS NULL 
#                  ORDER BY id DESC LIMIT 1""", (person_id,))
#     row = c.fetchone()
#     if row:
#         c.execute("UPDATE attendance SET check_out_time=?, check_out_location=? WHERE id=?", (now, location, row[0]))
#         conn.commit()
#         log_to_csv(name, "OUT", location)
#         print(f"[OUT] {name} at {now} from {location}")
#         exit_program()

# # --- Start Webcam ---
# cap = cv2.VideoCapture(0)

# print("🔵 Waiting for face detection...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         label, confidence = recognizer.predict(roi_gray)

#         if confidence < CONFIDENCE_THRESHOLD:
#             name = label_map.get(label, "Unknown")
#         else:
#             name = "Unknown"

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         if name != "Unknown":
#             person_id = get_person_id(name)
#             if not person_id:
#                 print(f"[ERROR] No ID found for {name}")
#                 continue

#             # Determine if IN or OUT
#             c.execute("""SELECT check_in_time, check_out_time FROM attendance 
#                          WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
#             record = c.fetchone()

#             if not record or record[1] is not None:
#                 log_check_in(person_id, name)
#             elif record[1] is None:
#                 log_check_out(person_id, name)

#     cv2.imshow("Face Attendance - Auto Exit", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("🔴 Quit key pressed.")
#         break

# exit_program()
























import cv2
import numpy as np
import sqlite3
import os
import csv
import sys
from datetime import datetime
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
import geocoder
from collections import deque, Counter

# --- Config ---
CONFIDENCE_THRESHOLD = 65 
REQUIRED_CONFIRMATIONS = 3
FRAME_WINDOW = 5
STILL_LIMIT = 5  # Frames with no movement allowed

# --- Anti-spoofing tracking variables ---
recent_predictions = deque(maxlen=FRAME_WINDOW)
prev_face = None
still_frame_count = 0

# --- Location Function ---
# def get_location():
#     try:
#         g = geocoder.ip('me')
#         city = g.city if g.city else "Unknown"
#         state = g.state if g.state else "Unknown"
#         return f"{city}, {state}"
#     except:
#         return "Unknown"

def get_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')
        loc = data.get('loc', '')  # lat,long if needed
        latlong = data.get('loc', '0,0')
        lat, lon = latlong.split(',')  
        return f"{city}, {region}, {country}"
    except Exception as e:
        print(f"[Location Error] {e}")
        return "Unknown"
    
# --- Google Sheet Logging (optional) ---
def log_to_gsheet(name, action, location):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('Attendance.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open("Attendance").sheet1
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        date_str = now.strftime('%d-%m-%Y') 
        time_str = now.strftime('%H:%M:%S') 
        # sheet.append_row([name, action, timestamp, location])
        sheet.append_row([name, action, date_str, time_str, location]) 
        print(f"[Google Sheet] {action} logged for {name} at {location}")
    except Exception as e:
        import traceback
        print("[Google Sheet Error]")
        traceback.print_exc()

# --- CSV Logging ---
def log_to_csv(name, action, location):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_path = 'attendance_log.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Action', 'Timestamp', 'Location'])
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, action, timestamp, location])
    print(f"[CSV] {action} logged for {name} at {location}")

# --- Exit cleanly ---
def exit_program():
    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    sys.exit(0) 

# --- Load Recognizer and Labels ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_map[int(label_id)] = name

# --- Database ---
conn = sqlite3.connect("attendance.db")
c = conn.cursor()

def get_person_id(name):
    c.execute("SELECT id FROM persons WHERE name=?", (name,))
    row = c.fetchone()
    return row[0] if row else None

def log_check_in(person_id, name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    location = get_location()
    c.execute("INSERT INTO attendance (person_id, check_in_time, check_in_location) VALUES (?, ?, ?)",
              (person_id, now, location))
    conn.commit()
    log_to_csv(name, "IN", location)
    log_to_gsheet(name, "IN", location) 
    print(f"[IN] {name} at {now} from {location}")
    exit_program()

def log_check_out(person_id, name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    location = get_location()
    c.execute("""SELECT id FROM attendance 
                 WHERE person_id=? AND check_out_time IS NULL 
                 ORDER BY id DESC LIMIT 1""", (person_id,))
    row = c.fetchone()
    if row:
        c.execute("UPDATE attendance SET check_out_time=?, check_out_location=? WHERE id=?",
                  (now, location, row[0]))
        conn.commit()
        log_to_csv(name, "OUT", location)
        log_to_gsheet(name, "OUT", location) 
        print(f"[OUT] {name} at {now} from {location}")
        exit_program()

# --- Webcam Start ---
cap = cv2.VideoCapture(0)
print("🟢 Please look at the webcam for IN/OUT Attendance...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    name = "Unknown"

    for (x, y, w, h) in faces:
        # 🛡️ Block small (spoofed) faces
        if w < 100 or h < 100:
            print("⚠️ Face too small — possible spoofing attempt.")
            continue

        # 🛡️ Motion check — detect if face hasn't moved
        if prev_face:
            px, py, pw, ph = prev_face
            dx = abs(x - px)
            dy = abs(y - py)
            dw = abs(w - pw)
            dh = abs(h - ph)
            movement = dx + dy + dw + dh 
            if movement < 15:
                still_frame_count += 1
            else:
                still_frame_count = 0
        prev_face = (x, y, w, h)

        if still_frame_count >= STILL_LIMIT:
            print("🛑 No motion detected — skipping to prevent spoofing.") 
            continue

        # Recognize face
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray) 

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label, "Unknown")
        else:
            name = "Unknown"

        recent_predictions.append(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Confirm consistent predictions
    if len(recent_predictions) == FRAME_WINDOW:
        most_common = Counter(recent_predictions).most_common(1)[0]
        if most_common[0] != "Unknown" and most_common[1] >= REQUIRED_CONFIRMATIONS:
            final_name = most_common[0]
            person_id = get_person_id(final_name)
            if not person_id:
                print(f"[ERROR] No ID found for {final_name}")
                exit_program()

            c.execute("""SELECT check_in_time, check_out_time FROM attendance 
                         WHERE person_id=? ORDER BY id DESC LIMIT 1""", (person_id,))
            record = c.fetchone()

            if not record or record[1] is not None:
                log_check_in(person_id, final_name)
            elif record[1] is None:
                log_check_out(person_id, final_name)

    cv2.imshow("Secure Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Exit requested.")
        break

exit_program() 