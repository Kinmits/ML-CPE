import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_models(models, X_test_scaled, y_test, output_dir='output'):
    """ประเมิน Accuracy Score และบันทึกรูป Confusion Matrix"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    accuracy_scores = {}
    predictions = {}
    
    for kernel, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores[kernel] = acc
        predictions[kernel] = y_pred
    
    # วาดและเซฟภาพ Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = sorted(y_test.unique())

    for ax, kernel in zip(axes, models.keys()):
        cm = confusion_matrix(y_test, predictions[kernel], labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'Kernel: {kernel.upper()}\nAccuracy: {accuracy_scores[kernel]*100:.2f}%')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'svm_confusion_matrices.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"[EVALUATION] Chart saved to {chart_path}")
    return accuracy_scores, predictions