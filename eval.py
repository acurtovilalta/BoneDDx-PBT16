import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score, f1_score,
    classification_report)
import torch
from utils import tensor_to_device
import os

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


def _eval_and_save(loader, split_name, model, device, num_classes, entities, base_out_dir, fold_idx):
    out = eval_main_loader(model, loader, device, num_classes, entities)
    # save report
    report_path = os.path.join(base_out_dir, f"fold{fold_idx+1}_{split_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(out["report"])
    return {k: v for k, v in out.items() if k != "report"}