"""Statistical toolkit fixed by the SAP."""
import numpy as np
import pandas as pd
from scipy import stats as ss

# Studentized range / sqrt(2) for alpha = 0.05 (Demsar 2006, Table 5a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
RNG_SEED = 20261018


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def cliff_label(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"W": np.nan, "p": 1.0}
    r = ss.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"W": float(r.statistic), "p": float(r.pvalue)}


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def boot_ci(x, n=10000, seed=RNG_SEED, alpha=0.05, stat=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = stat(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def friedman_nemenyi(df, block, treat, value, higher_better=True):
    """df long format. Returns (chi2, p, avg_ranks Series, CD, nemenyi p-matrix)."""
    wide = df.pivot_table(index=block, columns=treat, values=value, aggfunc="mean").dropna()
    sign = -1 if higher_better else 1
    ranks = wide.apply(lambda r: ss.rankdata(sign * r.values), axis=1, result_type="expand")
    ranks.columns = wide.columns
    chi2, p = ss.friedmanchisquare(*[wide[c].values for c in wide.columns])
    k, N = wide.shape[1], wide.shape[0]
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    import scikit_posthocs as sp
    nem = sp.posthoc_nemenyi_friedman(wide.values)
    nem.index = nem.columns = wide.columns
    return float(chi2), float(p), ranks.mean().sort_values(), float(cd), nem, N


def fmt_p(p):
    if p is None or np.isnan(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
