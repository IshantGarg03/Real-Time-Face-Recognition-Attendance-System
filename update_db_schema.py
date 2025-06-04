import sqlite3

conn = sqlite3.connect('attendance.db')
c = conn.cursor()

try:
    c.execute("ALTER TABLE attendance ADD COLUMN check_in_location TEXT")
except sqlite3.OperationalError:
    print("check_in_location column already exists.")

try:
    c.execute("ALTER TABLE attendance ADD COLUMN check_out_location TEXT")
except sqlite3.OperationalError:
    print("check_out_location column already exists.")

conn.commit()
conn.close()
print("Database schema updated.") 