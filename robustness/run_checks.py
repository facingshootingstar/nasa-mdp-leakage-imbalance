"""Additional analyses: pc2/mc2 robustness, duplicate-leak shares, inner-CV thresholds with resampling.
Uses the repo's core code; outputs go to robustness/out (existing files are skipped).
usage: python run_checks.py run <ds> | e1 <ds> | leak"""
import sys, pathlib, warnings
REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(REPO), str(REPO / "experiments")]
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import core.data as cd
from core.models import SEED
from core.metrics import threshold_metrics, rank_metrics
from run_p1_extra import fit_predict, group_folds, mcc_opt_threshold

warnings.filterwarnings("ignore")
ALL = cd.DATASETS + ["pc2", "mc2"]


def folds(y):  # identical to core.folds.get_folds, without writing fold files
    r = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    return [(tr, te, i // 5, i % 5) for i, (tr, te) in enumerate(r.split(np.zeros(len(y)), y))]


def leak():
    rows = []
    for ds in ALL:
        X, y, rep = cd.load(ds, clean=True, return_report=True)
        Xr, yr = cd.load(ds, clean=False); Xr = Xr.to_numpy()
        la, ld = [], []
        for tr, te, _, _ in folds(yr):
            s = {r.tobytes() for r in Xr[tr]}
            seen = np.array([r.tobytes() in s for r in Xr[te]])
            la.append(seen.mean()); ld.append(seen[yr[te] == 1].mean())
        rows.append({**rep, "leak_all": np.mean(la), "leak_def": np.mean(ld)})
    pd.DataFrame(rows).to_csv(OUT / "leak.csv", index=False); print(pd.DataFrame(rows).to_string())


def one(ds, kind, X, y, tr, te, rep, f, clf, rs):
    row = {"dataset": ds, "kind": kind, "repeat": rep, "fold": f, "clf": clf, "resampler": rs}
    if rs.endswith("+cv"):
        base = rs[:-3]
        oof = np.zeros(len(tr))
        for itr, ite in StratifiedKFold(3, shuffle=True, random_state=SEED + rep).split(X[tr], y[tr]):
            oof[ite] = fit_predict(clf, base, X[tr][itr], y[tr][itr], X[tr][ite], rep)
        grid = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
        t = float(grid[np.argmax([matthews_corrcoef(y[tr], (oof >= g).astype(int)) for g in grid])])
        p = fit_predict(clf, base, X[tr], y[tr], X[te], rep)
    elif rs == "thr_cv":
        t = mcc_opt_threshold(clf, X[tr], y[tr], rep); p = fit_predict(clf, "none", X[tr], y[tr], X[te], rep)
    elif rs == "thr_prior":
        t = y[tr].mean(); p = fit_predict(clf, "none", X[tr], y[tr], X[te], rep)
    else:
        t = 0.5; p = fit_predict(clf, rs, X[tr], y[tr], X[te], rep)
    tm = threshold_metrics(y[te], p, thr=t)
    tp = int(((p >= t) & (y[te] == 1)).sum()); fp = int(((p >= t) & (y[te] == 0)).sum())
    return {**row, "threshold": t, "tp": tp, "fp": fp, **tm, **rank_metrics(y[te], p)}


def run(ds, kind):
    out = OUT / f"{kind}_{ds}.csv"
    if out.exists():
        return
    clfs = ["LR", "RF", "LGBM"]
    if kind == "e1":
        X, y = cd.load(ds, clean=True); X = X.to_numpy()
        jobs = [(ds, "clean", X, y, tr, te, r, f, c, rs) for tr, te, r, f in folds(y) for c in clfs
                for rs in ("smote+cv", "smotetomek+cv", "rus+cv")]
    else:  # pc2/mc2: everything the 9-dataset files already hold for the other datasets
        Xc, yc = cd.load(ds, clean=True); Xc = Xc.to_numpy()
        Xr, yr = cd.load(ds, clean=False); Xr = Xr.to_numpy()
        jobs = [(ds, "clean", Xc, yc, tr, te, r, f, c, rs) for tr, te, r, f in folds(yc) for c in clfs
                for rs in ("none", "cw", "rus", "ros", "smote", "bsmote", "smotetomek", "thr_prior", "thr_cv")]
        jobs += [(ds, "raw", Xr, yr, tr, te, r, f, c, "none") for tr, te, r, f in folds(yr) for c in clfs]
        jobs += [(ds, "group", Xr, yr, tr, te, r, f, c, "none") for tr, te, r, f in group_folds(Xr, yr) for c in clfs]
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(one)(*j) for j in jobs)
    pd.DataFrame(res).to_csv(out, index=False); print(kind, ds, "done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "leak":
        leak()
    else:
        run(sys.argv[2], sys.argv[1])
