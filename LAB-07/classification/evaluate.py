import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def evaluate_performance(model, history, X_test, y_test, class_names, config_name):
    os.makedirs("output", exist_ok=True)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 [RESULT] {config_name} Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # วาดกราฟและ Confusion Matrix
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Accuracy ({config_name})')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Loss ({config_name})')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({config_name})')
    
    plt.tight_layout()
    plt.savefig(f"output/eval_{config_name}.png")
    plt.close()
    print(f"[INFO] Saved plots to output/eval_{config_name}.png")