"""Create the shared 5x5 repeated stratified folds once and reuse them everywhere."""
import json, pathlib
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from .models import SEED

FOLD_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "folds"


def get_folds(name, y, version="clean", n_splits=5, n_repeats=5):
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    f = FOLD_DIR / f"{name}_{version}.json"
    if f.exists():
        folds = json.loads(f.read_text())
        if folds[0]["n"] != len(y):
            raise ValueError(f"fold file {f} does not match data length")
        return [(np.array(d["train"]), np.array(d["test"]), d["repeat"], d["fold"]) for d in folds]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    out = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        out.append({"repeat": i // n_splits, "fold": i % n_splits, "n": len(y),
                    "train": tr.tolist(), "test": te.tolist()})
    f.write_text(json.dumps(out))
    return get_folds(name, y, version, n_splits, n_repeats)
