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


# ----------------------------------------------------------------------------- group-level (cluster) inference
# GA indicators of sentences in the same true event are dependent (they succeed or fail together), so the true
# event group is the resampling / randomisation unit: a cluster bootstrap and a group-level sign-flip test.
def _group_index(groups):
    keys, inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    return [np.where(inv == k)[0] for k in range(len(keys))]


def group_bootstrap_ci(correct, groups, weights=None, B=10000, alpha=0.05, seed=0):
    """Percentile CI of (weighted) accuracy, resampling whole true groups with replacement."""
    c = np.asarray(correct, float)
    w = np.ones_like(c) if weights is None else np.asarray(weights, float)
    idx = _group_index(groups)
    num = np.array([(c[i] * w[i]).sum() for i in idx])
    den = np.array([w[i].sum() for i in idx])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(idx), size=(B, len(idx)))
    stats = num[pick].sum(1) / den[pick].sum(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def group_paired_test(correct_a, correct_b, groups, weights=None, B=10000, seed=0):
    """Difference in (weighted) accuracy a - b with a group-bootstrap CI and a two-sided group-level
    sign-flip permutation p-value (under H0 the two methods' results are exchangeable within each true group)."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    idx = _group_index(groups)
    gd = np.array([((a[i] - b[i]) * w[i]).sum() for i in idx])
    den = np.array([w[i].sum() for i in idx])
    obs = gd.sum() / den.sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(idx), size=(B, len(idx)))
    boot = gd[pick].sum(1) / den[pick].sum(1)
    flips = rng.choice([-1.0, 1.0], size=(B, len(idx)))
    perm = (flips * gd).sum(1) / den.sum()
    p = float(((np.abs(perm) >= abs(obs) - 1e-12).sum() + 1) / (B + 1))
    return dict(diff=float(obs), ci=(float(np.quantile(boot, .025)), float(np.quantile(boot, .975))), p=min(1.0, p),
                groups_differing=int((gd != 0).sum()))
