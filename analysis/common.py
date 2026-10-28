"""Shared helpers for the analysis scripts: loading, plotting style, CD diagram, LaTeX tables."""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.stats import wilcoxon, holm, cliffs_delta, cliff_label, boot_ci, friedman_nemenyi, fmt_p

RAW = ROOT / "results" / "raw"
SUM = ROOT / "results" / "summary"; SUM.mkdir(parents=True, exist_ok=True)
# Okabe-Ito colour-blind safe palette
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "font.family": "DejaVu Sans"})

P1_COLS = ["dataset", "version", "repeat", "fold", "clf", "resampler", "leak_share", "n_test", "mcc", "gmean",
           "pd", "pf", "bacc", "auc", "prauc", "runtime_s", "mcc_unseen", "gmean_unseen"]


def load_p1():
    return pd.read_csv(RAW / "p1.csv", names=P1_COLS, skiprows=1)


def savefig(fig, paper, name):
    d = ROOT / f"paper{paper}" / "figs"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf"); fig.savefig(d / f"{name}.png")
    plt.close(fig)


def cd_diagram(avg_ranks, cd, title, ax=None, labels=None):
    """Demsar-style critical-difference diagram. avg_ranks: Series sorted ascending (1 = best)."""
    k = len(avg_ranks)
    names = list(avg_ranks.index); r = avg_ranks.values
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 0.55 + 0.26 * k))
    else:
        fig = ax.figure
    ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.set_yticks([])
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position("top"); ax.spines["top"].set_visible(True)
    ax.set_xticks(range(1, k + 1))
    half = (k + 1) // 2
    for i, (n, v) in enumerate(zip(names, r)):
        left = i < half
        y = 0.85 - 0.8 * ((i if left else k - 1 - i) + 1) / (half + 1)
        ax.plot([v, v], [1, y], color="k", lw=0.7)
        ax.plot([v, (1 if left else k)], [y, y], color="k", lw=0.7)
        lab = (labels or {}).get(n, n)
        ax.text((1 - 0.05) if left else (k + 0.05), y, f"{lab} ({v:.2f})",
                ha="right" if left else "left", va="center", fontsize=8, clip_on=False)
    # cliques: maximal groups whose rank range < cd
    cl = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for t, (a, b) in enumerate(cl):
        y = 0.93 - 0.05 * t
        ax.plot([r[a] - 0.03, r[b] + 0.03], [y, y], color=OI[3], lw=2.2)
    ax.plot([1, 1 + cd], [0.02, 0.02], color="k", lw=1)
    ax.text(1 + cd / 2, 0.05, f"CD = {cd:.2f}", ha="center", fontsize=7)
    ax.set_title(title, fontsize=9, pad=18)
    return fig


def pair_test(a, b):
    """Wilcoxon + Cliff's delta on paired dataset-level values; a is the treatment, b the reference."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    w = wilcoxon(a, b); d = cliffs_delta(a, b)
    lo, hi = boot_ci(a - b)
    return {"mean_a": a.mean(), "mean_b": b.mean(), "diff": (a - b).mean(), "ci_lo": lo, "ci_hi": hi,
            "W": w["W"], "p": w["p"], "delta": d, "mag": cliff_label(d), "n": len(a), "rrb": rank_biserial(a, b)}


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation (Kerby 2014): (sum of positive ranks - sum of negative ranks) / total.
    The paired effect size that matches the Wilcoxon signed-rank test (Cliff's delta treats the samples as unpaired)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[d != 0]
    if not len(d):
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def add_holm(df, by=None):
    if by is None:
        df["p_holm"] = holm(df.p.values)
    else:
        df["p_holm"] = df.groupby(by)["p"].transform(lambda s: holm(s.values))
    return df


def latex_table(df, path, caption, label, colfmt=None, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    colfmt = colfmt or ("l" * 1 + "r" * (len(cols) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", " & ".join(c if "$" in c else c.replace("_", r"\_") for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v).replace("_", r"\_").replace("%", r"\%"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def pfmt(p):
    return fmt_p(p)


def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return str(o)

    def keys(o):  # tuple keys (from MultiIndex to_dict) are not valid JSON keys
        if isinstance(o, (pd.Series, pd.DataFrame)):
            o = conv(o)
        if isinstance(o, dict):
            return {(":".join(map(str, k)) if isinstance(k, tuple) else k): keys(v) for k, v in o.items()}
        if isinstance(o, list):
            return [keys(v) for v in o]
        return o
    (SUM / name).write_text(json.dumps(keys(obj), indent=1, default=conv), encoding="utf-8")


def peq(p):
    """'<0.001' or '=0.012', for use after a 'p' in figure titles."""
    s = fmt_p(p)
    return s if s.startswith("<") else "=" + s
