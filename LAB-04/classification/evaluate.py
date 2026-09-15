import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from knn_tf import KNNTensorFlow

def evaluate_k_curve(X_train, y_train, X_test, y_test, max_k=15, output_path="outputs/01_k_curve.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    k_range = range(1, max_k + 1)
    accuracies = []
    
    for k in k_range:
        model = KNNTensorFlow(k=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, [a * 100 for a in accuracies], marker='o', color='#1f77b4', linewidth=2)
    plt.title("KNN Accuracy vs k Value", fontsize=14, fontweight='bold')
    plt.xlabel("k (Number of Neighbors)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    best_k = list(k_range)[np.argmax(accuracies)]
    return best_k, accuracies

def save_confusion_matrix(y_true, y_pred, output_path="outputs/02_confusion_matrix.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=['GKP', 'DEF', 'MID', 'FWD'])
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['GKP', 'DEF', 'MID', 'FWD'],
                yticklabels=['GKP', 'DEF', 'MID', 'FWD'])
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_predictions(player_names, y_true, y_pred, output_path="outputs/predictions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out = pd.DataFrame({
        'player_name': player_names,
        'true_position': y_true,
        'predicted_position': y_pred,
        'is_correct': y_true == y_pred
    })
    df_out.to_csv(output_path, index=False)