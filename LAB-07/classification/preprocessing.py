from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(X):
    print("[INFO] Preprocessing: Standardizing and reshaping for 1D CNN...")
    
    # ปรับสเกลตัวเลขให้สมดุล (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape สำหรับ Conv1D: (จำนวนแถว, จำนวนฟีเจอร์, 1 Channel)
    X_reshaped = np.expand_dims(X_scaled, axis=-1)
    
    print(f"[INFO] Data reshaped to: {X_reshaped.shape}")
    return X_reshaped, scaler