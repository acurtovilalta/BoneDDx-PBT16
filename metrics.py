import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix)
import torch
from utils import tensor_to_device



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

def eval_main_loader(model, loader, device, num_classes, entities):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels, aux_targets in loader:
            imgs, labels = imgs.to(device), tensor_to_device(labels, device)
            logits, _, _ = model(imgs)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=[str(e) for e in entities],
        digits=4, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "report": report,
    }