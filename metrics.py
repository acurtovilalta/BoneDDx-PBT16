import numpy as np
from sklearn.metrics import confusion_matrix

def invfreq_weighted_accuracy(y_true, y_pred, num_classes):
    """Inverse-frequency–weighted average of per-class recall."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    recall = np.zeros_like(support, dtype=float)
    mask = support > 0
    recall[mask] = tp[mask] / (support[mask] + 1e-10)
    inv_w = np.zeros_like(support, dtype=float)
    inv_w[mask] = 1.0 / support[mask]
    denom = inv_w.sum()
    if denom == 0:
        return 0.0
    inv_w /= denom
    return float((inv_w * recall).sum())
