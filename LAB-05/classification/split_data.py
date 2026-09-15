from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """แบ่งข้อมูลเป็น Train set และ Test set"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[SPLIT DATA] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test