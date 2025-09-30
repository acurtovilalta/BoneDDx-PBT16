import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class XrayDatasetMTL(Dataset):
    """
    Expects columns:
      - file_name (image filename)
      - entity    (int class index)
      - aux columns encoded as <aux>__encoded with -100 for missing
    Returns:
      img: Tensor [3,H,W]
      label: long []
      aux_targets: long tensor [num_aux]  (values in [0..C-1], -100 for missing)
    """
    def __init__(self, df, img_dir, aux_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.aux_cols = list(aux_cols)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, str(row['file_name']))).convert('RGB')
        if self.transform: img = self.transform(img)
        label = torch.tensor(int(row['entity']), dtype=torch.long)
        aux = [int(row.get(f"{c}__encoded", -100)) for c in self.aux_cols]
        aux = torch.tensor(aux, dtype=torch.long)
        return img, label, aux