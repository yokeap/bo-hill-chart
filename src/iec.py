"""IEC 60193 unit-quantity / affinity-law feature space.

Physics prior for experiment reduction. For a fixed-geometry machine (runner
diameter D) at fixed synchronous speed n, the IEC 60193 unit quantities are

    speed factor      n_ED = n D / sqrt(gH)
    discharge factor  Q_ED = Q / (D^2 sqrt(gH))
    power  factor     P_ED = P / (rho D^2 (gH)^{3/2})

with specific hydraulic energy E = gH. Hydraulic *similarity* states that
efficiency is a function of the unit operating point, eta = eta(n_ED, Q_ED),
independent of head. So operating points at different heads that share
(n_ED, Q_ED) are dynamically similar and have the same efficiency.

This dataset does not record D or n (grid-tied bulb turbine -> n constant, D
fixed), so absolute IEC factors are unknown. But n and D are *constants* here,
therefore

    n_ED  proportional_to  1/sqrt(E)         (varies only with head)
    Q_ED  proportional_to  Q/sqrt(E)

and the proportionality constants are irrelevant for GP modelling (features are
standardised) and for the similarity collapse (a common curve is a common curve
regardless of scale). We therefore work with these reduced forms.

Practical payoff we test: expressing the hill chart in unit space should
(a) make the efficiency surface smoother -> fewer experiments to reconstruct,
and (b) let a model predict *untested heads* by similarity (leave-one-head-out).
"""
import numpy as np

G = 9.81  # m/s^2


def unit_quantities(df):
    """Return reduced IEC unit quantities (proportional to n_ED, Q_ED, P_ED)."""
    H = df["Head"].values.astype(float)
    Q = df["Dischargem"].values.astype(float)
    E = G * H                      # specific hydraulic energy, J/kg
    sqrtE = np.sqrt(E)
    out = {
        "Q_ED": Q / sqrtE,         # ∝ discharge factor
        "n_ED": 1.0 / sqrtE,       # ∝ speed factor (monotone in 1/sqrt(H))
    }
    if "Generator power" in df.columns:
        out["P_ED"] = df["Generator power"].values.astype(float) / E ** 1.5
    return out


# named feature sets compared in the prototype
FEATURE_SETS = {
    "raw (Q,H,alpha)": ["Dischargem", "Head", "G/V degree"],
    "IEC (Q_ED,n_ED,alpha)": ["Q_ED", "n_ED", "G/V degree"],
    "IEC-similarity (Q_ED,alpha)": ["Q_ED", "G/V degree"],
}


def build_features(df, columns):
    """Assemble a feature matrix from raw and/or unit-quantity column names."""
    uq = unit_quantities(df)
    cols = []
    for c in columns:
        cols.append(uq[c] if c in uq else df[c].values.astype(float))
    return np.column_stack(cols)
