import cv2
import numpy as np
import tensorflow as tf

def preprocess_images(images, target_size=(224, 224)):
    print(f"[INFO] Preprocessing: Resizing to {target_size} and converting BGR to RGB...")
    processed_images = []
    
    for img in images:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to target size for VGG16 (224x224)
        img_resized = cv2.resize(img_rgb, target_size)
        processed_images.append(img_resized)
        
    processed_images = np.array(processed_images, dtype=np.float32)
    
    # Apply VGG16 specific preprocessing (ImageNet normalization)
    print("[INFO] Applying VGG16 preprocess_input...")
    processed_images = tf.keras.applications.vgg16.preprocess_input(processed_images)
    
    return processed_images