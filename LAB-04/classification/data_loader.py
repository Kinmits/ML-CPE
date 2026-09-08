import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_classification_data(csv_path="../fpl.csv", test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['minutes'] >= 180].copy()
    
    features = [
        'goals_scored', 'assists', 'clean_sheets', 'saves', 
        'influence', 'creativity', 'threat', 'ict_index', 
        'expected_goals', 'expected_assists', 'tackles', 
        'clearances_blocks_interceptions', 'recoveries', 'defensive_contribution'
    ]
    
    player_names = df_filtered['player_name'].values
    X = df_filtered[features].values
    y = df_filtered['position_name'].values
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, player_names, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "names_test": names_test
    }