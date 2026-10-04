from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    print("[INFO] Splitting data into Train, Validation, and Test sets...")
    
    # ดึง Test set ออกมาก่อน
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # ดึง Train และ Validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test