import os
import pandas as pd

from data_loader import load_dataset
from preprocessing import preprocess_data, scale_features
from split_data import split_data
from nn_model import train_nn_configurations
from evaluate import evaluate_nn_models

def main():
    print("==========================================")
    print(" LAB 6: Neural Network (NN) Pipeline")
    print("==========================================")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fpl_path = os.path.join(base_dir, 'fpl.csv')
    df = load_dataset(fpl_path)
    
    # 2. Preprocess Data
    X, y = preprocess_data(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Standardize Features
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    # 5. Define Configurations & Epochs
    configs = {
        'Config 1 (1 Layer: 32)': (32,),
        'Config 2 (1 Layer: 64)': (64,),
        'Config 3 (2 Layers: 64x32)': (64, 32)
    }
    epochs_list = [50, 100, 200, 500]
    
    # 6. Train Models
    trained_models = train_nn_configurations(X_train_scaled, y_train, configs, epochs_list)
    
    # 7. Evaluate Models
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    res_df = evaluate_nn_models(trained_models, X_train_scaled, y_train, X_test_scaled, y_test, output_dir)
    
    # Print Output Results
    print("\n==========================================")
    print(" Output 1: Accuracy & Loss Across Configurations and Epochs")
    print("==========================================")
    display_cols = ['Configuration', 'Max Epochs', 'Train Accuracy (%)', 'Test Accuracy (%)', 'Final Loss']
    print(res_df[display_cols].to_string(index=False))
    
    # Print Sample Predictions for Best Model
    best_row = res_df.loc[res_df['Test Accuracy (%)'].idxmax()]
    print("\n==========================================")
    print(f" Best Configuration: {best_row['Configuration']} with {best_row['Max Epochs']} Epochs")
    print(f" Highest Test Accuracy: {best_row['Test Accuracy (%)']}%")
    print("==========================================")
    
    sample_df = pd.DataFrame({
        'Actual Position': y_test.values[:15],
        'Predicted Position': best_row['Predictions'][:15]
    })
    print("\nSample Predictions (Top 15 Rows):")
    print(sample_df.to_string(index=False))

if __name__ == '__main__':
    main()