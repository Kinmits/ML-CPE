import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """เลือกสถิติผู้เล่นที่ใช้เป็น Features และ Target"""
    features = [
        'now_cost', 'total_points', 'goals_scored', 'assists', 'minutes', 
        'clean_sheets', 'goals_conceded', 'saves', 'yellow_cards', 'bonus',
        'influence', 'creativity', 'threat', 'ict_index',
        'expected_goals', 'expected_assists', 'tackles', 'recoveries'
    ]
    
    X = df[features].fillna(0)
    y = df['position_name']
    
    return X, y

def scale_features(X_train, X_test):
    """ปรับขนาดข้อมูลด้วย StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler