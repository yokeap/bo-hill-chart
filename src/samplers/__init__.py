"""Sampling strategies. Each is ``select(df, n_samples, rng) -> list[int]``
returning indices of chosen experiments. Fitting/scoring lives in metrics.evaluate.

Two families:
  space-filling / baseline : random, grid, lhs      (one-shot)
  sequential model-based   : bo_ucb, alm, alc        (fit GP each step)
"""
from .random_sampling import select_random
from .grid import select_grid
from .lhs import select_lhs
from .bo_ucb import select_bo_ucb
from .alm import select_alm
from .alc import select_alc

SAMPLERS = {
    "Random": select_random,
    "Grid": select_grid,
    "LHS": select_lhs,
    "BO-UCB": select_bo_ucb,
    "ALM": select_alm,
    "ALC": select_alc,
}
