"""Load and normalize the turbine performance dataset.

One CSV per head in ``bf-gv/`` named ``<head>m.csv``. Head comes from the
filename; rows with placeholder efficiency (>=1) or missing discharge are dropped.
"""
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ["Dischargem", "Head", "G/V degree"]
METRIC = "Overall Eff"


def load_all_data(folder_path="bf-gv"):
    """Return a tidy DataFrame of all experiments with a ``Head`` column."""
    folder = Path(folder_path)
    frames = []
    for file in sorted(folder.glob("*.csv")):
        head = float(file.stem.replace("m", ""))
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df["Head"] = head
        df = df[(df[METRIC] < 1) & (df["Dischargem"].notna())]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def feature_matrix(df):
    """Z-score-normalized feature matrix X and target y."""
    X = df[FEATURES].values.astype(float)
    y = df[METRIC].values.astype(float)
    Xn = (X - X.mean(axis=0)) / X.std(axis=0)
    return Xn, y
