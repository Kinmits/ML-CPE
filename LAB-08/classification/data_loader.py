import os
import cv2
import numpy as np
import json

def load_images_from_folder(dataset_path):
    print(f"[INFO] Loading images from {dataset_path}...")
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")
        
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"[INFO] Found classes: {class_names}")
    
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        count = 0
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Skipping corrupted file: {filename}")
                    continue
                images.append(img)
                labels.append(label)
                count += 1
            except Exception as e:
                print(f"[WARNING] Error reading {filename}: {e}")
        print(f"[INFO] Loaded {count} images for class '{class_name}'.")
        
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classes.json", "w") as f:
        json.dump(class_names, f)
        
    return images, np.array(labels), class_names