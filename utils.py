import random
import numpy as np
import pandas as pd
import torch

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