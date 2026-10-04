import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(csv_path="fpl.csv"):
    print(f"[INFO] Loading tabular dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # คัดเลือก 18 ฟีเจอร์ตัวเลข
    features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 
        'goals_conceded', 'saves', 'bonus', 'total_points', 
        'influence', 'creativity', 'threat', 'ict_index', 
        'now_cost', 'expected_goals', 'expected_assists', 
        'tackles', 'clearances_blocks_interceptions', 'recoveries'
    ]
    X = df[features].values
    
    # แปลงเป้าหมาย (GKP, DEF, MID, FWD) เป็นตัวเลข (0, 1, 2, 3)
    le = LabelEncoder()
    y = le.fit_transform(df['position_name'])
    class_names = le.classes_
    
    print(f"[INFO] Loaded {X.shape[0]} players. Classes: {class_names}")
    return X, y, class_names, features