import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import List, Dict, Optional
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import config



class HeartDiseaseNet(nn.Module):


    def __init__(
        self,
        n_features:    int  = config.N_FEATURES,
        n_classes:     int  = config.N_CLASSES,
        hidden_dims:   list = config.HIDDEN_DIMS,
        dropout_rates: list = config.DROPOUT_RATES,
    ):
        super().__init__()

        layers = []
        in_dim = n_features
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim >= 16:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Extrait les poids du modèle en numpy (pour l'échange FL)."""
        return [v.cpu().numpy() for v in self.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Charge les poids agrégés reçus du serveur FL."""
        keys = list(self.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.load_state_dict(state_dict, strict=True)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_one_round(
    model:         HeartDiseaseNet,
    X:             np.ndarray,
    y:             np.ndarray,
    epochs:        int,
    lr:            float,
    device:        torch.device,
    proximal_mu:   float = 0.0,
    global_params: Optional[List[torch.nn.Parameter]] = None,
) -> float:

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X).to(device)
    yt = torch.tensor(y).to(device)
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    total_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)

            if proximal_mu > 0.0 and global_params is not None:
                prox = sum(
                    ((p_l - p_g.to(device)) ** 2).sum()
                    for p_l, p_g in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (epochs * max(len(loader), 1))



def evaluate(
    model:  HeartDiseaseNet,
    X:      np.ndarray,
    y:      np.ndarray,
    device: torch.device,
) -> Dict:

    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X).to(device)
    yt = torch.tensor(y).to(device)

    with torch.no_grad():
        logits = model(Xt)
        loss   = criterion(logits, yt).item()
        preds  = logits.argmax(dim=1).cpu().numpy()

    return {
        "loss":      loss,
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y, preds, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y, preds, average="weighted", zero_division=0)),
        "preds":     preds,
        "labels":    y,
    }