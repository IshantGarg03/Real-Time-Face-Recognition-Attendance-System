# Code for clear the data in the database file

import sqlite3
import os

try:
    if os.path.exists("attendance.db"):
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        # Delete specific user (e.g., id=1)
        c.execute("DELETE FROM persons WHERE id = ?", (1,))  # Replace 1 with desired ID
        c.execute("DELETE FROM attendance WHERE person_id = ?", (1,))
        # Or delete all data
        # c.execute("DELETE FROM persons")
        # c.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        print("Database records deleted successfully.")
    else:
        print("attendance.db does not exist.")
except sqlite3.DatabaseError as e:
    print(f"Database error: {e}")
    os.remove("attendance.db")
    print("Deleted invalid attendance.db.")
except Exception as e:
    print(f"Error: {e}") 
