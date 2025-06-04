import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("Attendance.json", scope)
client = gspread.authorize(creds)

sheet = client.open("Attendance").sheet1
sheet.append_row(["Test User", "IN", "2025-06-04 12:00:00", "Office"])
print("✅ Data added to Google Sheet!")
