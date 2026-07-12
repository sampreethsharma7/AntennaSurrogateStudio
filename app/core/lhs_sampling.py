from typing import Optional

import numpy as np

try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except ImportError:
    _HAS_SCIPY_QMC = False


def generate_lhs_samples(bounds: dict[str, tuple[float, float]], n_samples: int, seed: Optional[int] = None) -> list[dict[str, float]]:
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if not bounds:
        raise ValueError("At least one parameter with bounds is required.")
    names = list(bounds.keys())
    lows = np.array([bounds[n][0] for n in names], dtype=float)
    highs = np.array([bounds[n][1] for n in names], dtype=float)
    if np.any(highs <= lows):
        raise ValueError("Each parameter's max bound must be greater than its min bound.")
    dimension = len(names)
    if _HAS_SCIPY_QMC:
        sampler = qmc.LatinHypercube(d=dimension, seed=seed)
        scaled = qmc.scale(sampler.random(n=n_samples), lows, highs)
    else:
        scaled = _fallback_lhs(dimension, n_samples, lows, highs, seed)
    return [dict(zip(names, row)) for row in scaled.tolist()]


def _fallback_lhs(dimension, n_samples, lows, highs, seed):
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, dimension))
    for d in range(dimension):
        cut_points = np.arange(n_samples) / n_samples
        offsets = rng.uniform(0, 1 / n_samples, n_samples)
        points = cut_points + offsets
        rng.shuffle(points)
        result[:, d] = lows[d] + points * (highs[d] - lows[d])
    return result
