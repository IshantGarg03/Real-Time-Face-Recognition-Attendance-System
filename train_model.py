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
