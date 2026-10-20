"""stats.py: paired significance tests for matched-seed comparisons."""
from __future__ import annotations
import math
import numpy as np


def paired_tests(x, y):
    """Two-sided paired t-test + Wilcoxon signed-rank on x - y (matched seeds)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    out = {"mean_diff": float(d.mean()), "n": n, "p_t": float("nan"),
           "p_wilcoxon": float("nan")}
    if n >= 2 and d.std(ddof=1) > 0:
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        try:
            from scipy import stats as st
            out["p_t"] = float(2 * st.t.sf(abs(t), n - 1))
            if n >= 5 and np.all(d != 0):
                out["p_wilcoxon"] = float(st.wilcoxon(x, y).pvalue)
        except ImportError:
            out["p_t"] = math.erfc(abs(t) / math.sqrt(2))  # normal approx
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    keys = [k for k in pvals if pvals[k] == pvals[k]]
    order = sorted(keys, key=lambda k: pvals[k])
    m, adj, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * pvals[k])
        adj[k] = min(1.0, running)
    for k in pvals:
        adj.setdefault(k, float("nan"))
    return adj