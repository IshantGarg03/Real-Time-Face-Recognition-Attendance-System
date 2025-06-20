# Code for clear the data or responses in the google sheet.

from oauth2client.service_account import ServiceAccountCredentials
import gspread

try:
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name("Attendance.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Attendance").sheet1
    # Get all records
    records = sheet.get_all_records()
    if not records:
        print("Google Sheet is already empty.")
    else:
        # Delete all rows except header (row 1)
        sheet.delete_rows(2, len(records) + 1)
        print("All Google Sheet data deleted successfully.")
except Exception as e:
    print(f"Error deleting Google Sheet data: {e}") 
