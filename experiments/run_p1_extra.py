"""Paper 1, revision analyses (deviation 28). One (part, dataset) per process; writes results/raw/p1x/<part>_<dataset>.csv.

parts
  group : raw data, 5x5 StratifiedGroupKFold with groups = identical metric vectors (no test row has a training twin)
  seen  : raw data, original random folds, no treatment; metrics on seen / unseen rows, prevalence, label consistency
  major : clean variant that keeps one row per conflicting group with the majority label (ties -> defective)
  thr   : clean data, no resampling, decision threshold moved (training prevalence; MCC-optimal on inner 3-fold OOF)
"""
import sys, pathlib, time, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from core.data import load, _raw
from core.folds import get_folds
from core.models import make, Prep, SEED
from core.metrics import threshold_metrics, rank_metrics
from run_p1 import sampler, RESAMPLERS, CLFS

warnings.filterwarnings("ignore")


def fit_predict(clf, rs, Xtr, ytr, Xte, rep):
    prep = Prep(clf); Ztr = prep.fit_transform(Xtr); Zte = prep.transform(Xte)
    s = sampler(rs, SEED + rep)
    if s is not None:
        Ztr, ytr = s.fit_resample(Ztr, ytr)
    m = make(clf, seed=SEED + rep, n_jobs=1, balanced=(rs == "cw")).fit(Ztr, ytr)
    return m.predict_proba(Zte)[:, 1]


def load_major(ds):
    """Majority-vote resolution of conflicting groups, then the other clean rules (missing, negative, duplicates)."""
    df = _raw(ds).copy()
    if df.shape[1] == 22:
        df.columns = [c if c == "__y" else c.lower().replace("locodeandcomment", "loccodeandcomment") for c in df.columns]
    feats = [c for c in df.columns if c != "__y"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(); df = df[(df[feats] >= 0).all(axis=1)]
    df["__y"] = (df.groupby(feats)["__y"].transform("mean") >= 0.5).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    return df[feats].astype(float), df["__y"].to_numpy()


def group_folds(X, y):
    gid = pd.factorize(pd.Series([r.tobytes() for r in X]))[0]
    out = []
    for rep in range(5):
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for f, (tr, te) in enumerate(sgk.split(X, y, gid)):
            out.append((tr, te, rep, f))
    return out


def mcc_opt_threshold(clf, X, y, rep):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED + rep).split(X, y):
        oof[te] = fit_predict(clf, "none", X[tr], y[tr], X[te], rep)
    grid = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
    return float(grid[np.argmax([matthews_corrcoef(y, (oof >= t).astype(int)) for t in grid])])


def main(part, ds):
    out = ROOT / "results" / "raw" / "p1x" / f"{part}_{ds}.csv"; out.parent.mkdir(exist_ok=True)
    if out.exists():
        return
    rows = []
    if part == "group":
        Xdf, y = load(ds, clean=False); X = Xdf.to_numpy()
        for tr, te, rep, f in group_folds(X, y):
            trset = {r.tobytes() for r in X[tr]}
            assert not any(r.tobytes() in trset for r in X[te])
            for clf in CLFS:
                for rs in RESAMPLERS:
                    p = fit_predict(clf, rs, X[tr], y[tr], X[te], rep)
                    rows.append({"dataset": ds, "repeat": rep, "fold": f, "clf": clf, "resampler": rs,
                                 "n_test": len(te), "prev_test": y[te].mean(),
                                 **threshold_metrics(y[te], p), **rank_metrics(y[te], p)})
    elif part == "seen":
        Xdf, y = load(ds, clean=False); X = Xdf.to_numpy()
        for tr, te, rep, f in get_folds(ds, y, "raw"):
            keys_tr = pd.Series([r.tobytes() for r in X[tr]])
            maj = pd.Series(y[tr]).groupby(keys_tr.values).mean()
            kte = [r.tobytes() for r in X[te]]
            seen = np.array([k in maj.index for k in kte])
            consistent = np.array([s and ((maj[k] >= 0.5) == bool(lab)) for k, s, lab in zip(kte, seen, y[te])])
            for clf in CLFS:
                p = fit_predict(clf, "none", X[tr], y[tr], X[te], rep)
                row = {"dataset": ds, "repeat": rep, "fold": f, "clf": clf, "n_test": len(te),
                       "share_seen": seen.mean(), "prev_full": y[te].mean(),
                       "prev_seen": y[te][seen].mean() if seen.any() else np.nan,
                       "prev_unseen": y[te][~seen].mean(), "consistent_of_seen": consistent[seen].mean() if seen.any() else np.nan,
                       "pos_unseen": int(y[te][~seen].sum())}
                for name, msk in (("full", np.ones(len(te), bool)), ("unseen", ~seen), ("seen", seen),
                                  ("unseen_or_consistent", ~seen | consistent)):
                    ok = msk.sum() and len(set(y[te][msk])) > 1
                    tm = threshold_metrics(y[te][msk], p[msk]) if ok else {}
                    rm = rank_metrics(y[te][msk], p[msk]) if ok else {}
                    for k in ("mcc", "gmean", "bacc"):
                        row[f"{k}_{name}"] = tm.get(k, np.nan)
                    row[f"auc_{name}"] = rm.get("auc", np.nan)
                rows.append(row)
    elif part == "major":
        Xdf, y = load_major(ds); X = Xdf.to_numpy()
        for tr, te, rep, f in get_folds(ds, y, "major"):
            for clf in CLFS:
                for rs in RESAMPLERS:
                    p = fit_predict(clf, rs, X[tr], y[tr], X[te], rep)
                    rows.append({"dataset": ds, "repeat": rep, "fold": f, "clf": clf, "resampler": rs,
                                 "n": len(y), "pos": int(y.sum()),
                                 **threshold_metrics(y[te], p), **rank_metrics(y[te], p)})
    elif part == "thr":
        Xdf, y = load(ds, clean=True); X = Xdf.to_numpy()
        for tr, te, rep, f in get_folds(ds, y, "clean"):
            for clf in CLFS:
                p = fit_predict(clf, "none", X[tr], y[tr], X[te], rep)
                for name, t in (("thr_prior", y[tr].mean()), ("thr_cv", mcc_opt_threshold(clf, X[tr], y[tr], rep))):
                    rows.append({"dataset": ds, "repeat": rep, "fold": f, "clf": clf, "resampler": name,
                                 "threshold": t, **threshold_metrics(y[te], p, thr=t), **rank_metrics(y[te], p)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(part, ds, "done", flush=True)


if __name__ == "__main__":
    t0 = time.time(); main(sys.argv[1], sys.argv[2]); print(f"{time.time() - t0:.0f}s")
