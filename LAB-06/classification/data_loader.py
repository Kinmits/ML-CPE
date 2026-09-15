import pandas as pd

def load_dataset(filepath):
    """โหลดข้อมูล FPL Dataset"""
    df = pd.read_csv(filepath)
    print(f"[DATA LOAD] Successfully loaded {len(df)} records from {filepath}")
    return df