"""
Data loading utilities for hill chart analysis
"""

import pandas as pd
from pathlib import Path


def load_all_data(folder_path='bf-gv'):
    """
    Load all CSV files and combine them with head information
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all experiments
    """
    folder = Path(folder_path)
    all_data = []
    
    csv_files = sorted(folder.glob('*.csv'))
    
    for file in csv_files:
        # Extract head from filename (e.g., '10m.csv' -> 10)
        head = float(file.stem.replace('m', ''))
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Add head column
        df['Head'] = head
        
        # Filter out invalid rows (efficiency > 1 indicates placeholder data)
        df = df[(df['Overall Eff'] < 1) & (df['Dischargem'].notna())]
        
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df