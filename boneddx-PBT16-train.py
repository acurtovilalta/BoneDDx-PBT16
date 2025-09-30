import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from losses import FocalLoss
from dataset import XrayDatasetMTL
from model import MultiTaskModel


# ==============================
# Utils
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    return torch.tensor(x, dtype=torch.long, device=device)


def print_per_class_counts(df, label_idx_to_name, header):
    print(f"\n{header}:")
    counts = df['entity'].value_counts().sort_index()
    for idx, cnt in counts.items():
        name = label_idx_to_name.get(idx, str(idx))
        print(f"  {name}: {cnt}")


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


# ==============================
# Label mapping for aux heads
# ==============================

def build_label_maps(df_train, aux_cols):
    """
    Create label maps for each aux column on TRAINALL to keep head dims stable.
    Classes: ['UNK', <sorted unique non-UNK values>]
    """
    ddx_label_maps, ddx_num_classes = {}, {}
    for c in aux_cols:
        vals = df_train[c].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        classes = ['UNK'] + [v for v in vals if v != 'UNK']
        lm = {v: i for i, v in enumerate(classes)}
        ddx_label_maps[c] = lm
        ddx_num_classes[c] = len(classes)
    return ddx_label_maps, ddx_num_classes


def encode_aux_columns(df, aux_cols, ddx_label_maps):
    df = df.copy()
    for c in aux_cols:
        lm = ddx_label_maps[c]
        def enc(v):
            if pd.isna(v): return -100
            v = str(v)
            return lm.get(v, lm['UNK'])
        df[f"{c}__encoded"] = df[c].map(enc)
    return df


# ==============================
# Evaluation
# ==============================

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


# ==============================
# One-fold train/eval
# ==============================

def train_one_fold(cfg, fold_idx, df_tr, df_va, df_test,
                   ddx_label_maps, ddx_num_classes, entities,
                   device, base_out_dir):
    """
    Train one CV fold; save best by VAL balanced accuracy (macro recall).
    """
    num_classes = len(entities)
    aux_cols = cfg["aux_cols"]

    # Main class weights from TRAIN fold (normalize)
    class_counts = df_tr['entity'].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights * (len(class_counts) / class_weights.sum())

    # Aux class weights (UNK=0)
    aux_class_weights = {}
    for c in aux_cols:
        ncls = ddx_num_classes[c]
        counts = np.zeros(ncls, dtype=np.float32)
        lm = ddx_label_maps[c]
        for k, idx in lm.items():
            if k == 'UNK':
                counts[idx] = 0.0
            else:
                counts[idx] = (df_tr[c].astype(str) == k).sum()
        counts = np.clip(counts, 1.0, None)
        w = 1.0 / counts
        w = w * (len(counts) / w.sum())
        w[lm['UNK']] = 0.0
        aux_class_weights[c] = torch.tensor(w, dtype=torch.float32)

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomResizedCrop(cfg["img_size"], scale=(cfg["aug_scale_min"], 1.0),
                                     ratio=(cfg["aug_ratio_min"], cfg["aug_ratio_max"])),
        transforms.RandomHorizontalFlip(p=cfg["aug_hflip_p"]),
        transforms.ColorJitter(brightness=cfg["aug_brightness"], contrast=cfg["aug_contrast"]),
        transforms.RandomAffine(degrees=cfg["aug_rotation"],
                                translate=(cfg["aug_translate"], cfg["aug_translate"]),
                                shear=cfg["aug_shear"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=cfg["aug_erasing_p"], scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & Loaders (we encoded aux on TRAINALL already)
    train_ds = XrayDatasetMTL(df_tr, cfg["img_dir"], aux_cols, transform=train_tfms)
    val_ds   = XrayDatasetMTL(df_va, cfg["img_dir"], aux_cols, transform=val_tfms)
    test_ds  = XrayDatasetMTL(df_test, cfg["img_dir"], aux_cols, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    # Model / Opt / Loss
    model = MultiTaskModel(
        num_entity_classes=num_classes,
        dropout=cfg["dropout"],
        img_emb_size=cfg["img_emb_size"],
        aux_cols=aux_cols,
        aux_num_classes=ddx_num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])

    ce_main = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=cfg["label_smoothing"])

    ce_aux = {}
    for c in aux_cols:
        w = aux_class_weights[c].to(device)
        if cfg["use_focal_aux"]:
            ce_aux[c] = FocalLoss(gamma=cfg["focal_gamma"], weight=w, reduction="mean")
        else:
            ce_aux[c] = nn.CrossEntropyLoss(weight=w, reduction="mean")

    # Bookkeeping
    best_val_bal_acc = -1.0
    epochs_no_improve = 0
    os.makedirs(base_out_dir, exist_ok=True)
    ckpt_path = os.path.join(base_out_dir, f"fold{fold_idx+1}_best.pt")

    # -------- Train loop --------
    for epoch in range(cfg["max_epochs"]):
        model.train()
        main_losses, aux_losses_log = [], []

        for imgs, labels, aux_targets in train_loader:
            imgs = imgs.to(device)
            labels = tensor_to_device(labels, device)
            aux_targets = aux_targets.to(device)

            optimizer.zero_grad()
            logits_main, logits_aux, img_emb = model(imgs)

            # main loss
            loss = ce_main(logits_main, labels)
            main_losses.append(loss.item())

            # cosine-annealed aux weight
            anneal = 0.5 * (1 + np.cos(
                np.pi * min(epoch, cfg["aux_anneal_epochs"]) / max(1, cfg["aux_anneal_epochs"])
            ))
            aux_w_eff = cfg["aux_weight"] * anneal

            # aux losses
            cur_aux_losses = []
            for i, c in enumerate(aux_cols):
                t = aux_targets[:, i]
                mask = t != -100
                if not mask.any(): continue

                if cfg["aux_detach"]:
                    # recompute logits on detached features to avoid pushing the trunk
                    logits_c = model.aux_heads[c](img_emb.detach())
                else:
                    logits_c = logits_aux[c]
                cur_aux_losses.append(ce_aux[c](logits_c[mask], t[mask]))

            if cur_aux_losses:
                aux_loss = torch.stack(cur_aux_losses).mean()
                aux_losses_log.append(aux_loss.item())
                loss = loss + aux_w_eff * aux_loss

            # step
            loss.backward()
            if cfg["grad_clip_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()

        # ---- Validation (main task) ----
        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for imgs, labels, aux_targets in val_loader:
                imgs, labels = imgs.to(device), tensor_to_device(labels, device)
                logits_main, _, _ = model(imgs)
                preds = torch.argmax(logits_main, dim=1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_bal_acc = recall_score(val_labels, val_preds, average='macro', zero_division=0)

        avg_main = float(np.mean(main_losses)) if main_losses else 0.0
        avg_aux  = float(np.mean(aux_losses_log)) if aux_losses_log else 0.0
        print(f"[Fold {fold_idx+1}] Epoch {epoch:03d} | train_main={avg_main:.4f} "
              f"| train_aux={avg_aux:.4f} | val_acc={val_acc:.4f} | val_bal_acc={val_bal_acc:.4f}")

        # Early stopping on balanced accuracy
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'entities': entities,
                'aux_cols': aux_cols,
                'ddx_label_maps': ddx_label_maps,
                'val_bal_acc': float(best_val_bal_acc),
                'saved_at': datetime.now().isoformat(timespec='seconds'),
                'config': cfg,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"[Fold {fold_idx+1}] Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # ---- Load best & final eval on VAL/TEST ----
    assert os.path.isfile(ckpt_path), f"Best checkpoint not found for fold {fold_idx+1}"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    def _eval_and_save(loader, split_name):
        out = eval_main_loader(model, loader, device, num_classes, entities)
        # save report
        report_path = os.path.join(base_out_dir, f"fold{fold_idx+1}_{split_name}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(out["report"])
        return {k: v for k, v in out.items() if k != "report"}

    val_metrics  = _eval_and_save(val_loader,  "VAL")
    test_metrics = _eval_and_save(test_loader, "TEST")

    return val_metrics, test_metrics, ckpt_path


# ==============================
# Cross-validation driver
# ==============================

def cross_validate(cfg):
    set_seed(cfg["split_seed"])
    # Device
    auto_dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"] if cfg["device"].startswith("cuda") and torch.cuda.is_available() else auto_dev)
    if device.type == "cpu":
        print("[INFO] CUDA not available; using CPU.")

    # Load labels file
    df_full = pd.read_excel(cfg["labels_path"])

    # Respect explicit TRAIN/TEST split (same as Methodology 1 variant)
    df_test = df_full[df_full['split'].astype(str).str.upper() == 'TEST'].copy()
    df_trainall = df_full[df_full['split'].astype(str).str.upper() == 'TRAIN'].copy()
    if df_trainall.empty:
        raise ValueError("Train split is empty. Ensure 'split' has 'TRAIN' and 'TEST'.")

    # Main label mapping fixed across folds
    # If 'entity' is already categorical names, we map to indices in sorted order
    entities = sorted(df_trainall['entity'].astype(str).unique())
    label_map = {name: idx for idx, name in enumerate(entities)}
    inv_label_map = {v: k for k, v in label_map.items()}

    df_trainall['entity'] = df_trainall['entity'].astype(str).map(label_map).astype(int)
    df_test['entity']     = df_test['entity'].astype(str).map(label_map).astype(int)

    # Aux maps from TRAINALL, then encode TRAINALL+TEST once
    ddx_label_maps, ddx_num_classes = build_label_maps(df_trainall, cfg["aux_cols"])
    df_trainall = encode_aux_columns(df_trainall, cfg["aux_cols"], ddx_label_maps)
    df_test     = encode_aux_columns(df_test,     cfg["aux_cols"], ddx_label_maps)

    # Output dir
    base_out = os.path.join(cfg["save_dir"], "folds_no_age_like_method1")
    os.makedirs(base_out, exist_ok=True)

    # 5-fold CV on TRAINALL
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=cfg["split_seed"])
    X_idx = np.arange(len(df_trainall))
    y_cls = df_trainall['entity'].to_numpy()

    # Collectors
    per_fold_rows = []
    all_val = { "accuracy": [], "balanced_accuracy": [], "recall_macro": [], "recall_weighted": [], "f1_macro": [] }
    all_test = { "accuracy": [], "balanced_accuracy": [], "recall_macro": [], "recall_weighted": [], "f1_macro": [] }

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_idx, y_cls)):
        print(f"\n========== Fold {fold_idx+1}/5 ==========")
        df_tr = df_trainall.iloc[tr_idx].copy()
        df_va = df_trainall.iloc[va_idx].copy()

        print_per_class_counts(df_tr, inv_label_map, header=f"Fold {fold_idx+1} - Train per-class counts")
        print_per_class_counts(df_va, inv_label_map, header=f"Fold {fold_idx+1} - Val  per-class counts")

        val_metrics, test_metrics, ckpt_path = train_one_fold(
            cfg, fold_idx, df_tr, df_va, df_test,
            ddx_label_maps, ddx_num_classes, entities,
            device, base_out
        )

        for k in all_val.keys():
            all_val[k].append(val_metrics[k])
            all_test[k].append(test_metrics[k])

        per_fold_rows.append({
            "fold": fold_idx+1,
            "best_ckpt": ckpt_path,
            **{f"val_{k}": val_metrics[k] for k in val_metrics.keys()},
            **{f"test_{k}": test_metrics[k] for k in test_metrics.keys()},
        })

    # Aggregate mean±sd
    def _mean_sd(vals):
        arr = np.asarray(vals, dtype=float)
        m = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return m, sd

    agg_rows = []
    for split_name, store in [("val", all_val), ("test", all_test)]:
        for metric, vals in store.items():
            m, sd = _mean_sd(vals)
            agg_rows.append({"split": split_name, "metric": metric, "mean": m, "sd": sd})

    per_fold_df = pd.DataFrame(per_fold_rows)
    agg_df = pd.DataFrame(agg_rows)

    # Save CSV + XLSX
    per_fold_csv = os.path.join(base_out, "cv_per_fold_metrics.csv")
    agg_csv      = os.path.join(base_out, "cv_aggregate_metrics.csv")
    per_fold_df.to_csv(per_fold_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)

    try:
        out_xlsx = os.path.join(base_out, "cv_summary.xlsx")
        with pd.ExcelWriter(out_xlsx) as writer:
            per_fold_df.to_excel(writer, index=False, sheet_name="per_fold")
            agg_df.to_excel(writer, index=False, sheet_name="aggregate")
        print(f"\nSaved summaries to:\n  {per_fold_csv}\n  {agg_csv}\n  {out_xlsx}")
    except Exception as e:
        print(f"\nSaved summaries to:\n  {per_fold_csv}\n  {agg_csv}\n(Excel export failed: {e})")


# ==============================
# Config
# ==============================

CFG = {
    "labels_path": "../tables/internal_data_split_clean_with_vlm_image_report.xlsx",
    "img_dir": "../../PrimaryBoneTumor16Class/Data/boundingboxes_all",
    "save_dir": "./CV-7-3Heads",

    # CV & device
    "split_seed": 42,
    "device": "cuda:0",

    # training control
    "max_epochs": 150,
    "patience": 50,
    "batch_size": 64, #32,
    "num_workers": 4,
    "img_size": 224,

    # model / loss / optim
    "img_emb_size": 512,
    "dropout": 0.2246108301788256,
    "label_smoothing": 0.07256863426168171,
    "lr": 0.0001,
    "weight_decay": 0.00001,
    "step_size": 7,
    "gamma": 0.8,
    "grad_clip_norm": 2.0,

    # auxiliary
    "aux_cols": ["lesion_pattern", "cortical_destruction", "periosteal_reaction"],
    "aux_weight": 0.19032676947212063,
    "aux_anneal_epochs": 40,
    "aux_detach": True, #False,         

    # focal loss for aux heads
    "use_focal_aux": True,
    "focal_gamma": 2.0,

    # augmentations
    "aug_scale_min": 0.8,
    "aug_ratio_min": 0.9,
    "aug_ratio_max": 1.1,
    "aug_hflip_p": 0.5,
    "aug_brightness": 0.4,
    "aug_contrast": 0.4,
    "aug_rotation": 10.0,
    "aug_translate": 0.05,
    "aug_shear": 5.0,
    "aug_erasing_p": 0.25,
}


if __name__ == "__main__":
    cross_validate(CFG)
