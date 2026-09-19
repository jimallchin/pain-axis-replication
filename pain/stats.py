"""Cluster bootstrap intervals and a cluster sign-flip test."""

import numpy as np


def cluster_bootstrap(values, clusters, n_boot=10000, seed=1337, stat=np.mean):
    """95% percentile interval for `stat`, resampling whole clusters with replacement."""
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    groups = [values[clusters == c] for c in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(values)), "lo": float(lo), "hi": float(hi),
            "n": int(len(values)), "clusters": int(len(groups))}  # fmt: skip


def cluster_bootstrap_frame(df, cluster_col, fn, n_boot=10000, seed=1337):
    """The same for a statistic of a whole data frame, e.g. a difference between arms."""
    keys = df[cluster_col].unique()
    parts = {k: g for k, g in df.groupby(cluster_col)}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    import pandas as pd

    for b in range(n_boot):
        pick = rng.choice(keys, len(keys))
        draws[b] = fn(pd.concat([parts[k] for k in pick], ignore_index=True))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": float(fn(df)), "lo": float(lo), "hi": float(hi), "clusters": int(len(keys))}


def cluster_sign_flip(values, clusters, n_perm=10000, seed=1337):
    """Two-sided p-value for mean zero, flipping the sign of whole clusters together."""
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    _, idx = np.unique(clusters, return_inverse=True)
    sums = np.bincount(idx, weights=values)
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    return float(((np.abs(signs @ sums) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)
