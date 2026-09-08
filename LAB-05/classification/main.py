import os
import pandas as pd

from data_load import load_dataset
from preprocess import preprocess_data, scale_features
from split_data import split_data
from svm_model import train_svm_models
from evaluate import evaluate_models

def main():
    print("==========================================")
    print(" LAB 5: Support Vector Machine (SVM) Pipeline")
    print("==========================================")
    
    # 1. Load Data (อ่านไฟล์ fpl.csv จากโฟลเดอร์นอก classification)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fpl_path = os.path.join(base_dir, 'fpl.csv')
    df = load_dataset(fpl_path)
    
    # 2. Preprocess Data
    X, y = preprocess_data(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Feature Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 5. Train Models
    kernels = ['linear', 'poly', 'rbf']
    models = train_svm_models(X_train_scaled, y_train, kernels)
    
    # 6. Evaluate Models (เซฟรูปไปที่โฟลเดอร์ output ข้างใน classification)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    accuracy_scores, predictions = evaluate_models(models, X_test_scaled, y_test, output_dir=output_dir)
    
    # Output 1: Print Accuracy Results
    print("\n==========================================")
    print(" Output 1: Accuracy Scores")
    print("==========================================")
    for kernel, score in accuracy_scores.items():
        print(f"SVM ({kernel.upper():<6} Kernel) Accuracy: {score * 100:.2f}%")
        
    # Output 2: Print Sample Predictions
    best_kernel = max(accuracy_scores, key=accuracy_scores.get)
    results_df = pd.DataFrame({
        'Actual Position': y_test.values,
        f'Predicted ({best_kernel.upper()})': predictions[best_kernel]
    }).reset_index(drop=True)
    
    print("\n==========================================")
    print(f" Output 2: Sample Predictions (Top 10 using {best_kernel.upper()} Kernel)")
    print("==========================================")
    print(results_df.head(10).to_string())

if __name__ == '__main__':
    main()