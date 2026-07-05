"""IEEE-style publication figures.

Conventions: Times serif, ~8 pt, single-column 3.5 in / double-column 7.16 in,
perceptually-uniform colormap (viridis), vector PDF output. Every 3D axes uses
the same viewing angle (VIEW) so surfaces are visually comparable.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .data_loader import METRIC

# ---- IEEE style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
COL1, COL2 = 3.5, 7.16          # IEEE column widths (inches)
VIEW = dict(elev=22, azim=-52)  # shared 3D view angle for ALL surface plots
CMAP = "viridis"

# stable per-method colors (ALC/ALM = greens = recommended; BO-UCB = red)
COLORS = {"ALC": "#1b7837", "ALM": "#5aae61", "BO-UCB": "#d6604d",
          "LHS": "#f4a582", "Grid": "#4393c3", "Random": "#7f7f7f"}


def _grid(q, h, z, n=120):
    qi = np.linspace(q.min(), q.max(), n)
    hi = np.linspace(h.min(), h.max(), n)
    Q, H = np.meshgrid(qi, hi)
    Z = griddata((q, h), z, (Q, H), method="cubic")
    return Q, H, Z


def _style3d(ax):
    ax.view_init(**VIEW)
    ax.set_xlabel(r"$Q$ (m$^3$/s)", labelpad=2)
    ax.set_ylabel(r"$H$ (m)", labelpad=2)
    ax.set_zlabel(r"$\eta$ (-)", labelpad=2)
    ax.tick_params(pad=1)


# ---- 3D surfaces ----------------------------------------------------------
def surface_3d(df, z, title, sampled_idx=None, bep=None, cmap=CMAP, err=False):
    q, h = df["Dischargem"].values, df["Head"].values
    Q, H, Z = _grid(q, h, z)
    fig = plt.figure(figsize=(COL1, COL1 * 0.85))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Q, H, Z, cmap=cmap, alpha=0.9, edgecolor="none", antialiased=True)
    if sampled_idx is not None:
        ax.scatter(q[sampled_idx], h[sampled_idx], np.asarray(z)[sampled_idx],
                   c="k", marker="o", s=10, depthshade=False, label="Sampled")
    if bep is not None:
        ax.scatter([bep["discharge"]], [bep["head"]], [bep["efficiency"]],
                   c="red", marker="s", s=26, edgecolors="k", linewidths=0.4,
                   depthshade=False, label="BEP")
    _style3d(ax)
    ax.set_title(title)
    if sampled_idx is not None or bep is not None:
        ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


# ---- 2D contours ----------------------------------------------------------
def contour_2d(df, z, title, sampled_idx=None, bep=None, cmap=CMAP, cbar_label=r"$\eta$"):
    q, h = df["Dischargem"].values, df["Head"].values
    Q, H, Z = _grid(q, h, z, n=150)
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.82))
    cf = ax.contourf(Q, H, Z, levels=20, cmap=cmap)
    cs = ax.contour(Q, H, Z, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%.2f")
    if sampled_idx is not None:
        ax.scatter(q[sampled_idx], h[sampled_idx], c="k", marker="o", s=12,
                   edgecolors="w", linewidths=0.4, label="Sampled", zorder=5)
    if bep is not None:
        ax.scatter([bep["discharge"]], [bep["head"]], c="red", marker="s", s=30,
                   edgecolors="k", linewidths=0.4, label="BEP", zorder=6)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlabel(r"$Q$ (m$^3$/s)")
    ax.set_ylabel(r"$H$ (m)")
    ax.set_title(title)
    if sampled_idx is not None or bep is not None:
        ax.legend(loc="upper left", framealpha=0.9)  # blank corner of the hill chart
    fig.tight_layout()
    return fig


METHOD_ORDER = ["ALM", "ALC", "Grid", "LHS", "Random", "BO-UCB"]


def all_methods_3d(df, preds, figsize=(COL2, COL2 * 0.62)):
    """2x3 grid of reconstructed 3D surfaces, one per method at a fixed budget.
    `preds` = {method: (y_pred, sampled_idx)}. Shared VIEW angle throughout."""
    q, h = df["Dischargem"].values, df["Head"].values
    fig = plt.figure(figsize=figsize)
    for k, mth in enumerate(METHOD_ORDER, 1):
        z, idx = preds[mth]
        Q, H, Z = _grid(q, h, z)
        ax = fig.add_subplot(2, 3, k, projection="3d")
        ax.plot_surface(Q, H, Z, cmap=CMAP, alpha=0.9, edgecolor="none", antialiased=True)
        ax.scatter(q[idx], h[idx], np.asarray(z)[idx], c="k", marker="o", s=5,
                   depthshade=False)
        ax.view_init(**VIEW)
        ax.set_title(mth, fontsize=8, pad=-2)
        ax.set_xlabel(r"$Q$ (m$^3$/s)", labelpad=-4, fontsize=6)
        ax.set_ylabel(r"$H$ (m)", labelpad=-4, fontsize=6)
        ax.set_zlabel(r"$\eta$ (-)", labelpad=-6, fontsize=6)
        ax.tick_params(labelsize=5, pad=-2)
    fig.tight_layout()
    return fig


def all_methods_error_3d(df, preds, figsize=(COL2, COL2 * 0.62)):
    """2x3 grid of 3D absolute-error surfaces |Delta eta|, one per method (the
    three-dimensional companion to all_methods_error)."""
    q, h = df["Dischargem"].values, df["Head"].values
    y = df[METRIC].values
    fig = plt.figure(figsize=figsize)
    for k, mth in enumerate(METHOD_ORDER, 1):
        z, idx = preds[mth]
        err = np.abs(y - np.asarray(z))
        Q, H, E = _grid(q, h, err)
        E = np.clip(E, 0.0, None)  # cubic interp can overshoot <0; |dEta| >= 0
        ax = fig.add_subplot(2, 3, k, projection="3d")
        ax.plot_surface(Q, H, E, cmap="magma", alpha=0.9, edgecolor="none",
                        antialiased=True)
        ax.scatter(q[idx], h[idx], err[idx], c="k", marker="o", s=5,
                   depthshade=False)
        ax.set_zlim(bottom=0)
        ax.view_init(**VIEW)
        ax.set_title(mth, fontsize=8, pad=-2)
        ax.set_xlabel(r"$Q$ (m$^3$/s)", labelpad=-4, fontsize=6)
        ax.set_ylabel(r"$H$ (m)", labelpad=-4, fontsize=6)
        ax.set_zlabel(r"$|\Delta\eta|$", labelpad=-6, fontsize=6)
        ax.tick_params(labelsize=5, pad=-2)
    fig.tight_layout()
    return fig


def _panel_grid(df, preds, cmap, cbar_label, bep=None, err_z=None,
                figsize=(COL2, COL2 * 0.58)):
    """Shared 2x3 contour grid over methods; `err_z` switches to |Delta eta| maps."""
    q, h = df["Dischargem"].values, df["Head"].values
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    cf = None
    for ax, mth in zip(axes.ravel(), METHOD_ORDER):
        z, idx = preds[mth]
        field = np.abs(df[METRIC].values - z) if err_z else z
        Q, H, Z = _grid(q, h, field, n=150)
        cf = ax.contourf(Q, H, Z, levels=18, cmap=cmap)
        ax.scatter(q[idx], h[idx], c="k", marker="o", s=8, edgecolors="w",
                   linewidths=0.3, zorder=5)
        if bep is not None:
            ax.scatter([bep["discharge"]], [bep["head"]], c="red", marker="s",
                       s=18, edgecolors="k", linewidths=0.4, zorder=6)
        ax.set_title(mth, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$Q$ (m$^3$/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H$ (m)")
    fig.colorbar(cf, ax=list(axes.ravel()), pad=0.02, label=cbar_label)
    return fig


def all_methods_2d(df, preds, bep=None, figsize=(COL2, COL2 * 0.58)):
    """2x3 contour grid of reconstructed hill charts, one per method."""
    return _panel_grid(df, preds, CMAP, r"$\eta$", bep=bep, figsize=figsize)


def all_methods_error(df, preds, figsize=(COL2, COL2 * 0.58)):
    """2x3 grid of absolute reconstruction error |Delta eta|, one per method."""
    return _panel_grid(df, preds, "magma", r"$|\Delta\eta|$", err_z=True,
                       figsize=figsize)


def uncertainty_error(df, sigma, err, idx, figsize=(COL2, COL2 * 0.42)):
    """Side-by-side GP posterior std and absolute error, to show they co-locate
    (supports posterior variance as a stopping signal)."""
    q, h = df["Dischargem"].values, df["Head"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, field, title, lab in [
        (a1, sigma, r"(a) Posterior uncertainty $\sigma$", r"$\sigma$"),
        (a2, err, r"(b) Absolute error $|\Delta\eta|$", r"$|\Delta\eta|$"),
    ]:
        Q, H, Z = _grid(q, h, field, n=150)
        cf = ax.contourf(Q, H, Z, levels=18, cmap="magma")
        ax.scatter(q[idx], h[idx], c="w", marker="o", s=10, edgecolors="k",
                   linewidths=0.4, zorder=5)
        cb = fig.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label(lab); cb.ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$Q$ (m$^3$/s)")
    a1.set_ylabel(r"$H$ (m)")
    fig.tight_layout()
    return fig


def sample_placement(df, placements, ground_z, figsize=(COL2, COL2 * 0.55)):
    """2x3 panel: where each method places samples at a low budget, over the
    ground-truth efficiency contour. `placements` = {method: idx list}."""
    q, h = df["Dischargem"].values, df["Head"].values
    Q, H, Z = _grid(q, h, ground_z, n=150)
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    for ax, (method, idx) in zip(axes.ravel(), placements.items()):
        ax.contourf(Q, H, Z, levels=18, cmap=CMAP, alpha=0.9)
        ax.scatter(q[idx], h[idx], c=COLORS[method], marker="o", s=14,
                   edgecolors="k", linewidths=0.4, zorder=5)
        ax.set_title(f"{method} ($n={len(idx)}$)")
    for ax in axes[-1]:
        ax.set_xlabel(r"$Q$ (m$^3$/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$H$ (m)")
    fig.tight_layout()
    return fig


# ---- learning curves ------------------------------------------------------
def _band(ax, sub, ycol, label, color, n_seeds):
    # standard error of the mean; lower bound clipped positive for the log axis
    sub = sub.sort_values("n")
    x, m = sub["n"], sub[f"{ycol}_mean"]
    sem = sub[f"{ycol}_std"].fillna(0) / np.sqrt(max(n_seeds, 1))
    lo = np.maximum(m - sem, m * 0.3)
    ax.plot(x, m, "-o", color=color, label=label, markersize=3)
    ax.fill_between(x, lo, m + sem, color=color, alpha=0.18, linewidth=0)


def goal_contrast(df, panels, bep_true, figsize=(COL2, COL2 * 0.46)):
    """Side-by-side reconstructions to show the two goals diverge.
    `panels` = list of (method_label, y_pred, sampled_idx, mae). Each panel marks
    the sampled points, the true BEP (red star), and the predicted BEP (black cross)."""
    q, h = df["Dischargem"].values, df["Head"].values
    # small margin so a BEP that lands on the domain boundary is not clipped
    dx, dy = 0.03 * (q.max() - q.min()), 0.03 * (h.max() - h.min())
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=True, sharey=True)
    for ax, (label, z, idx, mae) in zip(np.atleast_1d(axes), panels):
        Q, H, Z = _grid(q, h, z, n=150)
        cf = ax.contourf(Q, H, Z, levels=18, cmap=CMAP)
        ax.scatter(q[idx], h[idx], c="k", marker="o", s=12, edgecolors="w",
                   linewidths=0.4, label="Sampled", zorder=5)
        # true BEP = small red square (consistent with all other figures);
        # predicted BEP = a thin black plus, so the two are unambiguous.
        ax.scatter([bep_true["discharge"]], [bep_true["head"]], marker="s", s=30,
                   c="red", edgecolors="k", linewidths=0.4, label="True BEP",
                   zorder=6, clip_on=False)
        pk = int(np.argmax(z))
        ax.scatter([q[pk]], [h[pk]], marker="+", s=60, c="k", linewidths=1.1,
                   label="Predicted BEP", zorder=8, clip_on=False)
        ax.set_title(f"{label}  (MAE={mae:.4f})", fontsize=8)
        ax.set_xlabel(r"$Q$ (m$^3$/s)")
        ax.set_xlim(q.min() - dx, q.max() + dx)
        ax.set_ylim(h.min() - dy, h.max() + dy)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=6)  # blank corner here
    np.atleast_1d(axes)[0].set_ylabel(r"$H$ (m)")
    fig.colorbar(cf, ax=list(np.atleast_1d(axes)), pad=0.02, label=r"$\eta$")
    return fig


def metrics_bars(summary, n, n_seeds=10, figsize=(COL2, COL2 * 0.4)):
    """At a fixed budget n, bar-compare every method on the two headline metrics:
    reconstruction MAE (global) and BEP efficiency error (localization)."""
    order = ["ALM", "ALC", "Grid", "LHS", "Random", "BO-UCB"]
    sub = summary[summary["n"] == n].set_index("method").reindex(order)
    x = np.arange(len(order))
    cols = [COLORS[m] for m in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    for ax, key, lab, title in [
        (a1, "mae", "Reconstruction MAE", f"(a) Global reconstruction, $n={n}$"),
        (a2, "bep_eff_err", "BEP efficiency error", f"(b) BEP localization, $n={n}$"),
    ]:
        err = sub[f"{key}_std"].values / np.sqrt(max(n_seeds, 1))
        ax.bar(x, sub[f"{key}_mean"].values, yerr=err, color=cols,
               edgecolor="k", linewidth=0.5, error_kw=dict(lw=0.6))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(lab)
        ax.set_title(title, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":", linewidth=0.4)
    fig.tight_layout()
    return fig


def learning_curves(summary, n_seeds=10, figsize=(COL2, COL2 * 0.42)):
    """MAE (reconstruction) and BEP efficiency error vs. number of experiments.
    Bands are +/- 1 standard error of the mean over seeds."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    order = ["ALM", "ALC", "Grid", "LHS", "Random", "BO-UCB"]
    for method in order:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        _band(a1, sub, "mae", method, COLORS[method], n_seeds)
        _band(a2, sub, "bep_eff_err", method, COLORS[method], n_seeds)
    a1.set(xlabel="Number of experiments $n$", ylabel="Reconstruction MAE")
    a1.set_yscale("log")
    a1.set_title("(a) Full-map reconstruction", fontsize=8)
    a2.set(xlabel="Number of experiments $n$", ylabel="BEP efficiency error")
    a2.set_yscale("log")
    a2.set_title("(b) BEP localization", fontsize=8)
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3, linestyle=":", which="both", linewidth=0.4)
    # curves fall left-to-right, so the upper-right corner is the free space here
    a1.legend(loc="upper right", ncol=2, framealpha=0.9, handlelength=1.2,
              columnspacing=1.0)
    fig.tight_layout()
    return fig
