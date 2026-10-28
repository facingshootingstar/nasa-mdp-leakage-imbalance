"""Classification metrics."""
import numpy as np
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix)


def threshold_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    pd_ = tp / (tp + fn) if tp + fn else 0.0
    pf = fp / (fp + tn) if fp + tn else 0.0
    return {"mcc": matthews_corrcoef(y, yhat) if len(set(yhat)) > 1 or len(set(y)) > 1 else 0.0,
            "gmean": float(np.sqrt(pd_ * (1 - pf))), "pd": pd_, "pf": pf,
            "bacc": balanced_accuracy_score(y, yhat)}


def rank_metrics(y, p):
    if len(set(y)) < 2:
        return {"auc": np.nan, "prauc": np.nan}
    return {"auc": roc_auc_score(y, p), "prauc": average_precision_score(y, p)}
