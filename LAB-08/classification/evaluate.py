import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os

def evaluate_performance(model, history, X_test, y_test):
    print("\n[INFO] Evaluating model performance on Test Set...")
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/classes.json", "r") as f:
        class_names = json.load(f)
        
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 [RESULT] Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 1. Training History Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=300)
    plt.close()
    print("[INFO] Saved training history plot to outputs/training_history.png")
    
    # 2. Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close()
    print("[INFO] Saved confusion matrix to outputs/confusion_matrix.png")