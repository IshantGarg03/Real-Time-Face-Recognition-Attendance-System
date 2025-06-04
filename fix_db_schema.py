import sqlite3

conn = sqlite3.connect("attendance.db")  # Make sure this path matches your actual DB file
c = conn.cursor()

try:
    c.execute("ALTER TABLE attendance ADD COLUMN check_out_time TEXT")
    conn.commit()
    print("✅ check_out_time column added successfully.")
except sqlite3.OperationalError as e:
    print("❌ Error:", e)

conn.close()
