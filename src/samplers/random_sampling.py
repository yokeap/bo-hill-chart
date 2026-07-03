import numpy as np


def select_random(df, n_samples, rng):
    return [int(i) for i in rng.choice(len(df), size=n_samples, replace=False)]
