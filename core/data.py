"""Loading and cleaning of the NASA MDP datasets (OpenML copies)."""
import json, pathlib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(__file__).resolve().parent / ".cache"
DATA = ROOT / "data" / "nasa_mdp"  # shipped OpenML copies (see data/MANIFEST.sha256)
DATASETS = ["kc1", "kc2", "pc1", "jm1", "pc3", "pc4", "mc1", "mw1", "kc3"]
LOC_CANDIDATES = ["loc", "LOC_TOTAL", "LOC_EXECUTABLE", "lOCode"]


def _raw(name):
    if (DATA / f"{name}.csv").exists():
        return pd.read_csv(DATA / f"{name}.csv")
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    X, y = fetch_openml(name=name, version=1, return_X_y=True, as_frame=True,
                        parser="auto", data_home=str(CACHE / "openml"))
    df = X.copy()
    df["__y"] = y.astype(str).str.lower().isin(["true", "yes", "1"]).astype(int)
    df.to_pickle(f)
    return df


def loc_column(df):
    for c in LOC_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("no LOC column")


def load(name, clean=True, return_report=False):
    df = _raw(name).copy()
    if df.shape[1] == 22:  # 21-metric schema: harmonise case/typos across kc1, kc2, pc1, jm1
        df.columns = [c if c == "__y" else c.lower().replace("locodeandcomment", "loccodeandcomment")
                      for c in df.columns]
    feats = [c for c in df.columns if c != "__y"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    report = {"dataset": name, "n_raw": len(df), "pos_raw": int(df["__y"].sum())}
    if clean:
        n0 = len(df)
        df = df.dropna()
        report["missing_removed"] = n0 - len(df)
        n0 = len(df)
        df = df[(df[feats] >= 0).all(axis=1)]
        report["negative_removed"] = n0 - len(df)
        n0 = len(df)
        # conflicting: identical features, different labels -> drop every member
        g = df.groupby(feats, dropna=False)["__y"].transform("nunique")
        df = df[g == 1]
        report["conflicting_removed"] = n0 - len(df)
        n0 = len(df)
        df = df.drop_duplicates()
        report["duplicate_removed"] = n0 - len(df)
    else:
        df = df.fillna(df[feats].median())
    df = df.reset_index(drop=True)
    report["n_final"] = len(df)
    report["pos_final"] = int(df["__y"].sum())
    X = df[feats].astype(float)
    y = df["__y"].to_numpy()
    if return_report:
        return X, y, report
    return X, y
