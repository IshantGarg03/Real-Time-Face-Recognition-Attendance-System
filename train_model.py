# # import cv2
# # import os
# # import numpy as np

# # def train_model():
# #     faces = []
# #     labels = []
# #     label_map = {}
# #     current_label = 0

# #     base_path = "face_data"  # Change this to your base folder containing subfolders for each person

# #     for person_name in os.listdir(base_path):
# #         person_path = os.path.join(base_path, person_name)

# #         for date_folder in os.listdir(person_path):
# #             folder_path = os.path.join(person_path, date_folder)

# #             for img_file in os.listdir(folder_path):
# #                 img_path = os.path.join(folder_path, img_file)
# #                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# #                 if img is None:
# #                     continue

# #                 faces.append(img)
# #                 labels.append(current_label)

# #         label_map[current_label] = person_name
# #         current_label += 1

# #     # Create recognizer and train
# #     recognizer = cv2.face.LBPHFaceRecognizer_create()
# #     recognizer.train(faces, np.array(labels))

# #     recognizer.save("trained_model.yml")

# #     with open("labels.txt", "w") as f:
# #         for label_id, name in label_map.items():
# #             f.write(f"{label_id},{name}\n")

# #     print("✅ Model trained and saved.")

# # if __name__ == "__main__":
# #     train_model()





















# import cv2
# import os
# import numpy as np

# def train_model():
#     faces = []
#     labels = []
#     label_map = {}
#     current_label = 0

#     base_path = "face_data"

#     for person_name in os.listdir(base_path):
#         person_path = os.path.join(base_path, person_name)

#         if not os.path.isdir(person_path):
#             print(f"[SKIP] Not a directory: {person_path}")
#             continue

#         for date_folder in os.listdir(person_path):
#             folder_path = os.path.join(person_path, date_folder)

#             if not os.path.isdir(folder_path):
#                 print(f"[SKIP] Not a directory: {folder_path}")
#                 continue

#             for img_file in os.listdir(folder_path):
#                 img_path = os.path.join(folder_path, img_file)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is None:
#                     print(f"[SKIP] Could not read image: {img_path}")
#                     continue

#                 faces.append(img)
#                 labels.append(current_label)

#         label_map[current_label] = person_name
#         current_label += 1

#     if not faces:
#         print("❌ No valid training images found.")
#         return

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, np.array(labels))
#     recognizer.save("trained_model.yml")

#     with open("labels.txt", "w") as f:
#         for label_id, name in label_map.items():
#             f.write(f"{label_id},{name}\n")

#     print("✅ Model trained and saved.")

# if __name__ == "__main__":
#     train_model()






















# import cv2
# import os
# import numpy as np

# data_dir = 'face_data'
# face_data = []
# labels = []
# label_map = {}

# print("🔁 Scanning folders...")

# for folder in os.listdir(data_dir):
#     folder_path = os.path.join(data_dir, folder)
#     if not os.path.isdir(folder_path):
#         continue

#     # Expecting format: 0_Ishant
#     try:
#         label_id, name = folder.split('_', 1)
#         label_id = int(label_id)
#         label_map[label_id] = name
#     except:
#         print(f"[SKIP] Invalid folder format: {folder}")
#         continue

#     for file in os.listdir(folder_path):
#         if file.endswith(".jpg") or file.endswith(".png"):
#             file_path = os.path.join(folder_path, file)
#             img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"[SKIP] Unable to read image: {file_path}")
#                 continue
#             face_data.append(img)
#             labels.append(label_id)

# if not face_data:
#     print("❌ No valid training images found.")
#     exit()

# # Train model
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.train(face_data, np.array(labels))
# recognizer.save('trained_model.yml')
# print("✅ Model trained and saved as trained_model.yml")

# # Save labels
# with open("labels.txt", "w") as f:
#     for label_id, name in label_map.items():
#         f.write(f"{label_id},{name}\n")

# print("✅ labels.txt saved.")


























import cv2
import os
import numpy as np
from PIL import Image
import sqlite3

# --- Configuration ---
dataset_path = "face_data"
model_path = "trained_model.yml"
label_file = "labels.txt"

# --- Step 1: Create label mapping from folder names ---
def get_label_mapping():
    label_map = {}
    folders = sorted(os.listdir(dataset_path))
    for folder in folders:
        if os.path.isdir(os.path.join(dataset_path, folder)):
            label_id, name = folder.split("_", 1)
            label_map[int(label_id)] = name
    return label_map

# --- Step 2: Load training data from all folders ---
def load_training_data():
    faces = []
    labels = []
    label_map = get_label_mapping()

    for label_id, name in label_map.items():
        person_dir = os.path.join(dataset_path, f"{label_id}_{name}")
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = Image.open(img_path).convert('L')  # grayscale
            img_np = np.array(img, 'uint8')
            faces.append(img_np)
            labels.append(label_id)
    
    return faces, labels, label_map

# --- Step 3: Train the model ---
def train_and_save_model(faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)
    print(f"✅ Model trained and saved to {model_path}")

# --- Step 4: Save label map to labels.txt ---
def save_labels_txt(label_map):
    with open(label_file, "w") as f:
        for label_id, name in label_map.items():
            f.write(f"{label_id},{name}\n")
    print(f"✅ Label file saved to {label_file}")

# --- Run Everything ---
if __name__ == "__main__":
    print("🔄 Loading training data...")
    faces, labels, label_map = load_training_data()

    if len(faces) == 0:
        print("❌ No training images found. Please add data in 'face_data/'")
    else:
        train_and_save_model(faces, labels)
        save_labels_txt(label_map)
        print(f"🎉 Training complete! {len(set(labels))} people trained.") 

























# import cv2
# import os 
# import numpy as np
# from PIL import Image

# dataset_path = "face_data"
# model_path = "trained_model.yml"
# label_file = "labels.txt"

# def get_label_mapping():
#     label_map = {}
#     folders = sorted(os.listdir(dataset_path))
#     for folder in folders:
#         if os.path.isdir(os.path.join(dataset_path, folder)):
#             label_id, name = folder.split("_", 1)
#             label_map[int(label_id)] = name
#     return label_map

# def load_training_data():
#     faces = []
#     labels = []
#     label_map = get_label_mapping()
#     for label_id, name in label_map.items():
#         person_dir = os.path.join(dataset_path, f"{label_id}_{name}")
#         for filename in os.listdir(person_dir):
#             img_path = os.path.join(person_dir, filename)
#             img = Image.open(img_path).convert('L')
#             img_np = np.array(img, 'uint8')
#             faces.append(img_np)
#             labels.append(label_id)
#     return faces, labels, label_map

# def train_and_save_model():
#     faces, labels, label_map = load_training_data()
#     if len(faces) == 0:
#         print("No training images found.")
#         return False
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, np.array(labels))
#     recognizer.save(model_path)
#     with open(label_file, "w") as f:
#         for label_id, name in label_map.items():
#             f.write(f"{label_id},{name}\n")
#     print("Model trained and saved.")
#     return True
