from sklearn.model_selection import train_test_split
import numpy as np
import os

def split_and_save_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    print("[INFO] Splitting dataset into Train, Validation, and Test sets...")
    
    # Split out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split remaining into train and validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"[INFO] Train set shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"[INFO] Validation set shape: {X_val.shape}, labels: {y_val.shape}")
    print(f"[INFO] Test set shape: {X_test.shape}, labels: {y_test.shape}")
    
    # Save arrays to outputs/ directory
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/X_train.npy", X_train)
    np.save("outputs/X_val.npy", X_val)
    np.save("outputs/X_test.npy", X_test)
    np.save("outputs/y_train.npy", y_train)
    np.save("outputs/y_val.npy", y_val)
    np.save("outputs/y_test.npy", y_test)
    np.save("outputs/labels.npy", y)
    print("[INFO] Saved dataset arrays (.npy) to outputs/ directory.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test