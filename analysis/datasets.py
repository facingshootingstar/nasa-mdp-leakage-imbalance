"""Dataset characteristics and cleaning report (shared by all papers)."""
from common import *
from core.data import load, DATASETS, loc_column

rows = []
for d in DATASETS:
    Xr, yr = load(d, clean=False)
    X, y, rep = load(d, return_report=True)
    rows.append({**rep, "n_metrics": X.shape[1], "loc_col": loc_column(X),
                 "pos_rate_raw": rep["pos_raw"] / rep["n_raw"], "pos_rate_final": rep["pos_final"] / rep["n_final"],
                 "removed_share": 1 - rep["n_final"] / rep["n_raw"]})
df = pd.DataFrame(rows)
df.to_csv(SUM / "datasets.csv", index=False)
print(df.to_string())
