import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    """
    ResNet18 backbone -> image_mlp -> main classifier + parallel aux heads.
    """
    def __init__(self, num_entity_classes=16, dropout=0.3, img_emb_size=512,
                 aux_cols=None, aux_num_classes=None):
        super().__init__()
        aux_cols = aux_cols or []
        aux_num_classes = aux_num_classes or {}

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone

        self.image_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, img_emb_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(img_emb_size, num_entity_classes)

        self.aux_cols = list(aux_cols)
        self.aux_heads = nn.ModuleDict()
        for c in self.aux_cols:
            ncls = aux_num_classes[c]
            self.aux_heads[c] = nn.Linear(img_emb_size, ncls)

    def forward(self, x):
        feats = self.cnn(x)
        img_emb = self.image_mlp(feats)
        logits_main = self.classifier(img_emb)
        logits_aux = {c: self.aux_heads[c](img_emb) for c in self.aux_cols}
        return logits_main, logits_aux, img_emb

