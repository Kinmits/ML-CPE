import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_clustering_data(csv_path="../fpl.csv"):
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return {
        "X_scaled": X_scaled,
        "player_names": player_names,
        "raw_df": df_filtered
    }