import os
for root, dirs, files in os.walk("face_data"):
    print(f" Directory: {root}") 
    for file in files:
        print(f"  - {file}") 