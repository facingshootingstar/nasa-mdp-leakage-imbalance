"""Paper 1: data cleaning x resampling. Per-fold results -> results/raw/p1.csv"""
import sys, pathlib, time, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from core.data import load, DATASETS
from core.folds import get_folds
from core.models import make, Prep, SEED
from core.metrics import threshold_metrics, rank_metrics

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "raw" / "p1.csv"
RESAMPLERS = ["none", "cw", "rus", "ros", "smote", "bsmote", "smotetomek"]
CLFS = ["LR", "RF", "LGBM"]


def sampler(name, seed):
    k = 5
    return {"rus": RandomUnderSampler(random_state=seed), "ros": RandomOverSampler(random_state=seed),
            "smote": SMOTE(random_state=seed, k_neighbors=k),
            "bsmote": BorderlineSMOTE(random_state=seed, k_neighbors=k),
            "smotetomek": SMOTETomek(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=k))}.get(name)


def run_fold(ds, version, X, y, tr, te, rep, fold):
    rows = []
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    # leakage: test rows whose metric vector also occurs in training fold
    trset = {r.tobytes() for r in Xtr}
    seen = np.array([r.tobytes() in trset for r in Xte])
    for clf in CLFS:
        prep = Prep(clf); Ztr = prep.fit_transform(Xtr); Zte = prep.transform(Xte)
        for rs in RESAMPLERS:
            t0 = time.perf_counter()
            Zr, yr = Ztr, ytr
            s = sampler(rs, SEED + rep)
            if s is not None:
                Zr, yr = s.fit_resample(Ztr, ytr)
            m = make(clf, seed=SEED + rep, n_jobs=1, balanced=(rs == "cw")).fit(Zr, yr)
            p = m.predict_proba(Zte)[:, 1]
            row = {"dataset": ds, "version": version, "repeat": rep, "fold": fold, "clf": clf,
                   "resampler": rs, "leak_share": seen.mean(), "n_test": len(te),
                   **threshold_metrics(yte, p), **rank_metrics(yte, p),
                   "runtime_s": time.perf_counter() - t0}
            if version == "raw":
                u = ~seen
                mu = threshold_metrics(yte[u], p[u]) if u.sum() and len(set(yte[u])) > 1 else {"mcc": np.nan, "gmean": np.nan}
                row["mcc_unseen"] = mu["mcc"]; row["gmean_unseen"] = mu["gmean"]
            rows.append(row)
    return rows


def main():
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT); done = set(zip(old.dataset, old.version, old.repeat, old.fold))
    for ds in DATASETS:
        for version in ["clean", "raw"]:
            Xdf, y = load(ds, clean=(version == "clean"))
            X = Xdf.to_numpy()
            folds = [f for f in get_folds(ds, y, version) if (ds, version, f[2], f[3]) not in done]
            if not folds:
                continue
            t0 = time.time()
            res = Parallel(n_jobs=-1)(delayed(run_fold)(ds, version, X, y, tr, te, r, f) for tr, te, r, f in folds)
            df = pd.DataFrame([r for rr in res for r in rr])
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(ds, version, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
