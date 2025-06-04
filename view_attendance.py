import sqlite3

conn = sqlite3.connect('attendance.db')
c = conn.cursor()

print("📋 Persons Table:")
for row in c.execute("SELECT * FROM persons"):
    print(row)

print("\n🕒 Attendance Records:")
for row in c.execute("SELECT * FROM attendance"):
    print(row)

conn.close() 