import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_nn_models(trained_models, X_train_scaled, y_train, X_test_scaled, y_test, output_dir):
    """ประเมิน Accuracy, Loss Curve และเซฟรูปภาพลงโฟลเดอร์ output"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    evaluation_results = []
    
    for config_name, epoch_dict in trained_models.items():
        for epochs, model in epoch_dict.items():
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            loss = model.loss_
            
            evaluation_results.append({
                'Configuration': config_name,
                'Hidden Layers': str(model.hidden_layer_sizes),
                'Max Epochs': epochs,
                'Actual Iterations': model.n_iter_,
                'Train Accuracy (%)': round(train_acc * 100, 2),
                'Test Accuracy (%)': round(test_acc * 100, 2),
                'Final Loss': round(loss, 4),
                'Model': model,
                'Predictions': test_pred
            })
            
    res_df = pd.DataFrame(evaluation_results)
    
    # 1. วาดกราฟเปรียบเทียบ Loss & Accuracy ตามจำนวน Epochs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for config_name in trained_models.keys():
        subset = res_df[res_df['Configuration'] == config_name]
        plt.plot(subset['Max Epochs'], subset['Test Accuracy (%)'], marker='o', label=config_name)
    plt.title('Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for config_name in trained_models.keys():
        subset = res_df[res_df['Configuration'] == config_name]
        plt.plot(subset['Max Epochs'], subset['Final Loss'], marker='s', linestyle='--', label=config_name)
    plt.title('Final Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nn_epochs_comparison.png'), dpi=300)
    plt.close()
    
    # 2. วาด Confusion Matrix สำหรับโมเดลที่ดีที่สุด
    best_row = res_df.loc[res_df['Test Accuracy (%)'].idxmax()]
    best_model = best_row['Model']
    best_preds = best_row['Predictions']
    
    plt.figure(figsize=(6, 5))
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Best NN Confusion Matrix\n{best_row['Configuration']} ({best_row['Max Epochs']} Epochs)")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_nn_confusion_matrix.png'), dpi=300)
    plt.close()
    
    return res_df