"""Scoring: global reconstruction accuracy and BEP localization.

BEP is read off the GP prediction over the *full* dataset, not just the
sampled rows -- otherwise any method that happens to sample the true peak
scores a trivially perfect BEP error.
"""
import numpy as np
from .data_loader import feature_matrix, FEATURES, METRIC
from .gp_model import train_gp_model


def calculate_metrics(y_true, y_pred):
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
        "r2": float(1 - ss_res / ss_tot),
    }


def ground_truth_bep(df):
    i = df[METRIC].values.argmax()
    row = df.iloc[i]
    return {"discharge": float(row["Dischargem"]), "head": float(row["Head"]),
            "efficiency": float(row[METRIC]), "gv": float(row["G/V degree"])}


def evaluate(df, indices):
    """Fit a GP on the sampled rows, predict all points, score reconstruction + BEP."""
    Xn, y = feature_matrix(df)
    gp = train_gp_model(Xn[indices], y[indices])
    y_pred = gp.predict(Xn)

    out = calculate_metrics(y, y_pred)

    # BEP from the full-grid prediction (predicted peak vs. true peak)
    bep_idx = int(np.argmax(y_pred))
    true = ground_truth_bep(df)
    row = df.iloc[bep_idx]
    out["bep_error"] = {
        "discharge": abs(float(row["Dischargem"]) - true["discharge"]),
        "head": abs(float(row["Head"]) - true["head"]),
        "efficiency": abs(float(y_pred[bep_idx]) - true["efficiency"]),
    }
    out["y_pred"] = y_pred
    return out
