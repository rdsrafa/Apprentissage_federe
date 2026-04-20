"""
Dirichlet/model/network.py
─────────────────────────────────────────────────────────────────────────────
Architecture du réseau de neurones et utilitaires d'entraînement/évaluation.

Architecture : MLP avec BatchNorm et Dropout.
Choisi plutôt que les modèles à base d'arbres car :
  1. Les paramètres peuvent être moyennés entre clients (FedAvg)
  2. Compatible avec la régularisation proximale de FedProx
  3. Capacité suffisante pour 13 features tabulaires sans overfitting

Ce module est partagé entre le serveur (initialisation, évaluation)
et chaque client hôpital (entraînement local).
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# Racine du projet = deux niveaux au-dessus de Dirichlet/model/
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

from federated_health import config


# ── Modèle ────────────────────────────────────────────────────────────────────

class HeartDiseaseNet(nn.Module):
    """
    MLP compact pour la classification binaire tabulaire (Cleveland dataset).

    Architecture par défaut :
        Input(13) → Linear(64) → BN → ReLU → Dropout(0.3)
                  → Linear(32) → BN → ReLU → Dropout(0.2)
                  → Linear(16) → ReLU
                  → Linear(2)  → logits

    BatchNorm  : stabilise l'entraînement sur des mini-batches hétérogènes.
    Dropout    : régularise pour éviter l'overfitting sur de petits datasets locaux.
    Xavier     : initialisation qui accélère la convergence dès le round 1.
    """

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
        """Initialisation Xavier uniforme pour toutes les couches linéaires."""
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
    """
    Entraînement local pour un round FL (variante FedProx).

    Le terme proximal de FedProx contraint les mises à jour locales
    à rester proches du modèle global, évitant la dérive des clients
    sous des données non-IID :

        L_local = L_CE + (μ/2) * ||w_local - w_global||²

    Sans terme proximal (μ=0), c'est un entraînement FedAvg classique.

    Paramètres
    ----------
    model          : modèle local à entraîner en place
    X, y           : données d'entraînement locales de l'hôpital
    epochs         : nombre d'époques locales pour ce round
    lr             : learning rate (décroissant, schedulé par le serveur)
    device         : cpu ou cuda
    proximal_mu    : force de régularisation FedProx (0 = FedAvg)
    global_params  : copie figée des paramètres du modèle global

    Retourne
    --------
    float : perte d'entraînement moyenne pour ce round
    """
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

            # Terme proximal FedProx
            if proximal_mu > 0.0 and global_params is not None:
                prox = sum(
                    ((p_l - p_g.to(device)) ** 2).sum()
                    for p_l, p_g in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            # Gradient clipping : stabilise l'entraînement sur petits batches
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (epochs * max(len(loader), 1))


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model:  HeartDiseaseNet,
    X:      np.ndarray,
    y:      np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Évaluation complète sur un ensemble held-out.

    Retourne accuracy, precision/recall/F1 pondérés, loss, prédictions brutes.
    Moyenne pondérée pour gérer le déséquilibre de classes.
    """
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