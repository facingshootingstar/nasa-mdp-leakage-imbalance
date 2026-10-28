# Duplicate leakage and class-imbalance treatments in NASA MDP defect prediction — reproduction package

Data and code that reproduce every number, table and figure of the paper (Paper 1): how cross-fold
duplicate leakage inflates defect-prediction scores on the NASA MDP datasets, and how imbalance
treatments (class weights, RUS, ROS, SMOTE, Borderline-SMOTE, SMOTE-Tomek) compare with simple
threshold rules once the data are cleaned.

## Contents

| Path | What it is |
|---|---|
| `data/nasa_mdp/` | The 11 NASA MDP datasets as fetched from OpenML (`fetch_openml(name, version=1)`), saved as CSV. 9 are used in the main study (kc1, kc2, pc1, jm1, pc3, pc4, mc1, mw1, kc3); pc2 and mc2 are only used in the robustness check. `data/MANIFEST.sha256` holds their hashes. |
| `core/` | Loading and cleaning (`data.py`), fixed folds (`folds.py`), learners (`models.py`), metrics, statistics. Seed 20261018. |
| `experiments/` | `run_p1.py` (main grid: 9 datasets × clean/raw × 3 learners × 7 treatments × 5×5 CV), `run_p1_extra.py` (group CV, majority-vote cleaning, threshold rules, seen/unseen split). |
| `analysis/` | `datasets.py`, `p1.py`: statistics, LaTeX tables and figures from the raw results. |
| `results/folds/` | The fixed 5×5 stratified folds for every dataset and version. |
| `reference/` | The outputs of the original run: raw per-fold results, `summary/p1.json`, `paper1/tables.tex`, figures. |
| `robustness/` | Additional analyses (pc2/mc2, duplicate-leak shares, inner-CV thresholds combined with resampling, Spearman mechanism, false alarms per detected defect), with its per-fold outputs in `robustness/out/`. |
| `repro.py` | One entry point that runs everything and compares it with `reference/`. |

## Run

Python 3.12 (original runs: 3.12.10, Windows 11).

```bash
pip install -r requirements.txt
```

```bash
python repro.py check
```

```bash
python repro.py fast
```

`fast` takes the shipped raw results (`reference/raw/`), re-runs the full statistical analysis and checks
that every JSON value, the LaTeX tables and the figures match `reference/`. The report is written to
`out/report_p1_fast.md` and ends with `RESULT: ALL REPRODUCED` or a list of differences.

```bash
python repro.py full -j 8
```

`full` re-trains every model from `data/` with the fixed folds, then analyses and compares. It takes hours.
Tree learners may differ in the last digits on another OS or CPU.

### Additional analyses (`robustness/`)

```bash
python robustness/summarize.py
```

This prints the values from the shipped outputs. To recompute the outputs from scratch, delete
`robustness/out/` and run `python robustness/run_checks.py leak`, then `run <ds>` for pc2 and mc2 and `e1 <ds>`
for all 11 datasets. False alarms per detected defect are computed as pf·(1−π)/(pd·π) from mean pd, pf and
the clean defect rate π of each dataset.

## Cleaning rules

`core/data.py:load` drops, in order: rows with missing values, rows with negative metric values, every
member of a group of identical metric vectors that carry conflicting labels, and exact duplicates. On the
9 main datasets this removes 11,294 rows, 9,889 of them duplicates.
