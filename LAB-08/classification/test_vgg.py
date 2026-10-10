import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os

def test_random_images(model_path="outputs/vgg_model.keras"):
    print(f"\n[INFO] Testing random images using model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found.")
        return
        
    model = tf.keras.models.load_model(model_path)
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")
    
    with open("outputs/classes.json", "r") as f:
        class_names = json.load(f)
        
    num_samples = min(4, len(X_test))
    indices = random.sample(range(len(X_test)), num_samples)
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        img_preprocessed = X_test[idx]
        true_label = class_names[y_test[idx]]
        
        pred_prob = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)
        pred_idx = np.argmax(pred_prob)
        pred_label = class_names[pred_idx]
        confidence = np.max(pred_prob) * 100
        
        # Reverse VGG preprocessing roughly for display
        img_display = img_preprocessed.copy()
        img_display += np.array([103.939, 116.779, 123.68], dtype=np.float32)
        img_display = np.clip(img_display, 0, 255).astype(np.uint8)
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_display)
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", color=color, fontsize=11)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("outputs/prediction_sample.png", dpi=300)
    #plt.show()
    print("[INFO] Saved prediction sample plot to outputs/prediction_sample.png")

if __name__ == "__main__":
    test_random_images()