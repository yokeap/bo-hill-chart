"""Physics-informed GP: a hydraulic loss model as the prior mean.

A plain GP starts from a flat prior mean and must learn the whole efficiency
hill from data. We instead give it a physical prior: the fundamental turbine
relation  P_hyd = rho g Q H  and  eta = P_out / P_hyd  means the *power loss*
P_loss = P_hyd - P_out is a smooth, low-order function of the operating point.
We fit a hydraulic loss model

    P_loss ~= a Q^2 + b Q + c + d H          (friction ~Q^2, plus offsets)

from the sampled points (linear in the coefficients -> needs only ~4 points),
convert it to a mean efficiency  m(Q,H) = (P_hyd - P_loss_model)/P_hyd, and let
the GP model the residual eta - m. The physics carries the gross shape; the GP
corrects the rest, so accuracy at very low sample counts improves.

Unlike the affinity-law collapse (see src/iec.py), this needs no similarity
assumption, so it applies to fixed-speed machines.
"""
import numpy as np
from .data_loader import feature_matrix, METRIC
from .gp_model import train_gp_model
from .metrics import calculate_metrics, ground_truth_bep

G = 9.81      # m/s^2
RHO = 1000.0  # kg/m^3
N_COEF = 4    # loss-model parameters
RIDGE = 0.3   # regularisation on loss coefficients; essential at low n to stop
              # the loss model overfitting a handful of points (see physics_reduction.py)


def hydraulic_power_kW(df):
    return RHO * G * df["Dischargem"].values * df["Head"].values / 1000.0


def _loss_design(df):
    Q = df["Dischargem"].values
    H = df["Head"].values
    return np.column_stack([Q ** 2, Q, H, np.ones_like(Q)])


def physics_mean_predictor(df_train, ridge=RIDGE):
    """Fit the (ridge-regularised) loss model on sampled rows; return eta-mean(df_eval).
    Falls back to a constant mean if too few points to fit."""
    P_hyd = hydraulic_power_kW(df_train)
    P_loss = P_hyd - df_train["Generator power"].values
    if len(df_train) < N_COEF:
        const = float(df_train[METRIC].mean())
        return lambda df_eval: np.full(len(df_eval), const)
    A = _loss_design(df_train)
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(N_COEF), A.T @ P_loss)

    def mean_eta(df_eval):
        Phy = hydraulic_power_kW(df_eval)
        eta = (Phy - _loss_design(df_eval) @ coef) / Phy
        return np.clip(eta, 0.01, 0.999)
    return mean_eta


def evaluate_physics(df, indices):
    """Residual GP on top of the physical mean. Mirrors metrics.evaluate output."""
    Xn, y = feature_matrix(df)
    mean_fn = physics_mean_predictor(df.iloc[list(indices)])
    m_all = mean_fn(df)
    gp = train_gp_model(Xn[indices], y[indices] - m_all[indices])
    y_pred = m_all + gp.predict(Xn)

    out = calculate_metrics(y, y_pred)
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
