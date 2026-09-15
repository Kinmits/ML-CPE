import pandas as pd

def load_dataset(filepath='../fpl.csv'):
    """โหลดข้อมูล FPL Dataset"""
    df = pd.read_csv(filepath)
    print(f"[DATA LOAD] Loaded {len(df)} rows from {filepath}")
    return df