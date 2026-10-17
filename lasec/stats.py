"""Uncertainty and significance testing for per-item correctness indicators."""
import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(correct, weights=None, B=2000, alpha=0.05, seed=0):
    """Percentile CI of (weighted) accuracy, resampling unique items."""
    c = np.asarray(correct, float)
    w = np.ones_like(c) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(B, len(c)))
    stats = (c[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def cluster_bootstrap_ci(correct_by_group, B=2000, alpha=0.05, seed=0):
    """CI resampling whole groups (e.g. flights); each group is an array of per-occurrence indicators."""
    groups = [np.asarray(g, float) for g in correct_by_group]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        stats.append(v.mean())
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap(correct_a, correct_b, weights=None, B=5000, seed=0):
    """Two-sided paired bootstrap test on the difference in (weighted) accuracy a - b."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    d_obs = ((a - b) * w).sum() / w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = ((a[idx] - b[idx]) * w[idx]).sum(1) / w[idx].sum(1)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=float(d_obs), ci=(float(np.quantile(d, .025)), float(np.quantile(d, .975))), p=float(min(1.0, p)))


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on unique-item correctness (discordant pairs only)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only=n10, b_only=n01, p=float(p))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
