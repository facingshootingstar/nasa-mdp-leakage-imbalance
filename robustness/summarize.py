"""Summarise the additional analyses. Reads reference/raw + robustness/out."""
import pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[1] / "reference" / "raw"
OUT = pathlib.Path(__file__).resolve().parent / "out"
D9 = ["kc1", "kc2", "pc1", "jm1", "pc3", "pc4", "mc1", "mw1", "kc3"]
leak = pd.read_csv(OUT / "leak.csv").set_index("dataset")
prev = leak.pos_final / leak.n_final
rng = np.random.default_rng(1)


def boot(d, B=10000):
    d = np.asarray(d); m = rng.choice(d, (B, len(d))).mean(1)
    return f"{d.mean():+.3f} [{np.quantile(m, .025):+.3f}, {np.quantile(m, .975):+.3f}]"


# --- clean-data rows for all treatments, all 11 datasets
p1 = pd.read_csv(REPO / "p1.csv")
clean = [p1[p1.version == "clean"].drop(columns=["version"])]
clean += [pd.read_csv(REPO / "p1x" / f"thr_{d}.csv").assign(dataset=d) for d in D9]
for d in ["pc2", "mc2"]:
    f = OUT / f"run_{d}.csv"
    if f.exists():
        r = pd.read_csv(f); clean.append(r[r.kind == "clean"])
for f in OUT.glob("e1_*.csv"):
    clean.append(pd.read_csv(f))
C = pd.concat(clean, ignore_index=True)
M = C.groupby(["dataset", "resampler"])[["mcc", "gmean", "pd", "pf"]].mean()
mcc = M.mcc.unstack(); gm = M.gmean.unstack()

print("== mean MCC over 9 datasets (mean of dataset means)")
print(mcc.loc[D9].mean().round(3).sort_values().to_string())
for a, b in [("smote+cv", "thr_cv"), ("smotetomek+cv", "thr_cv"), ("rus+cv", "thr_cv"), ("thr_cv", "smote")]:
    if a in mcc and b in mcc:
        d = (mcc[a] - mcc[b]).loc[D9]
        print(f"{a} - {b} (n=9): {boot(d)}  higher on {(d > 0).sum()}/9")

d11 = [d for d in D9 + ["pc2", "mc2"] if d in mcc.index and not np.isnan(mcc.loc[d, "smote"])]
print(f"\n== n={len(d11)}")
print("SMOTE - none MCC:", boot((mcc.smote - mcc.none).loc[d11]))
print("G-Mean SMOTE > none on", ((gm.smote - gm.none).loc[d11] > 0).sum(), "/", len(d11))
if "smote+cv" in mcc:
    dd = [d for d in d11 if not np.isnan(mcc.loc[d].get("smote+cv", np.nan))]
    print(f"SMOTE+CV - CV (n={len(dd)}):", boot((mcc["smote+cv"] - mcc.thr_cv).loc[dd]))

print("\n== false alarms per detected defect = mean pf (1-pi) / (mean pd pi), median over 9 datasets")
fa = (M.pf.unstack().mul(1 - prev, axis=0)) / (M.pd.unstack().mul(prev, axis=0))
print(fa.loc[D9].median().round(2).sort_values().to_string())
print("mc1 range over treatments:", fa.loc["mc1"].round(1).min(), "-", fa.loc["mc1"].round(1).max())
print(fa.loc["mc1"].round(1).to_string())

# --- mechanism: inflation (RF, none) = standard raw CV - group CV
raw = p1[(p1.version == "raw") & (p1.resampler == "none")]
grp = pd.concat([pd.read_csv(REPO / "p1x" / f"group_{d}.csv") for d in D9])
grp = grp[grp.resampler == "none"]
for d in ["pc2", "mc2"]:
    f = OUT / f"run_{d}.csv"
    if f.exists():
        r = pd.read_csv(f); raw = pd.concat([raw, r[r.kind == "raw"]]); grp = pd.concat([grp, r[r.kind == "group"]])
for clf in ["RF", "LR", "LGBM"]:
    inf = (raw[raw.clf == clf].groupby("dataset").mcc.mean() - grp[grp.clf == clf].groupby("dataset").mcc.mean()).dropna()
    L = leak.loc[inf.index]
    print(f"\n{clf} inflation (n={len(inf)}): rho(def leak)={spearmanr(inf, L.leak_def)[0]:.2f} "
          f"rho(all leak)={spearmanr(inf, L.leak_all)[0]:.2f}")
    if clf == "RF":
        print(inf.round(3).to_string())
