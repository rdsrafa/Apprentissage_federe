"""
federated_health/data/dataset.py
─────────────────────────────────────────────────────────────────────────────
Chargement, preprocessing et partitionnement non-IID (Dirichlet)
du Cleveland Heart Disease Dataset (303 patients, 13 features).

Décisions de design :
  - StandardScaler fitté UNIQUEMENT sur le train (pas de data leakage)
  - Partitionnement Dirichlet : simule l'hétérogénéité réelle entre hôpitaux
  - Chaque partition est auto-portante (indices + métadonnées)

En déploiement réel, ce module serait remplacé par des data loaders
spécifiques à chaque hôpital — le serveur n'y accède jamais.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# Remonte à la racine du projet pour trouver config.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config


# ── Métadonnées des features ──────────────────────────────────────────────────

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age":      "Âge (années)",
    "sex":      "Sexe (1=H, 0=F)",
    "cp":       "Type de douleur thoracique (0–3)",
    "trestbps": "Pression artérielle au repos (mmHg)",
    "chol":     "Cholestérol sérique (mg/dl)",
    "fbs":      "Glycémie à jeun > 120 mg/dl (binaire)",
    "restecg":  "Résultats ECG au repos (0–2)",
    "thalach":  "Fréquence cardiaque maximale atteinte",
    "exang":    "Angine induite par l'exercice (binaire)",
    "oldpeak":  "Dépression ST induite par l'exercice",
    "slope":    "Pente du segment ST au pic d'effort",
    "ca":       "Nombre de vaisseaux majeurs colorés (0–4)",
    "thal":     "Thalassémie (1=normal, 2=défaut fixe, 3=défaut réversible)",
}

CLASS_NAMES = ["Pas de maladie", "Maladie cardiaque"]


# ── Classe principale ─────────────────────────────────────────────────────────

class HeartDataset:
    """
    Charge et prépare le Cleveland Heart Disease Dataset.

    Attributs
    ---------
    X_train, y_train : np.ndarray
        Features et labels d'entraînement (normalisés).
    X_test, y_test : np.ndarray
        Features et labels de test (jamais vus pendant l'entraînement).
    scaler : StandardScaler
        Scaler fitté sur train uniquement.
    n_features : int
    n_classes  : int
    """

    def __init__(self, path: str = None):
        if path is None:
            # Cherche heart.csv à la racine du projet
            path = os.path.join(ROOT, "heart.csv")

        df = pd.read_csv(path)

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["target"].values.astype(np.int64)

        # Split stratifié : conserve les proportions de classes dans les deux splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y,
        )

        # Normalisation : fit sur train uniquement → pas de fuite vers le test
        self.scaler  = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)

        self.n_features = X.shape[1]
        self.n_classes  = len(np.unique(y))

        self._print_summary(df, y)

    def _print_summary(self, df: pd.DataFrame, y: np.ndarray):
        print(f"[HeartDataset] {len(df)} patients chargés — "
              f"{self.X_train.shape[0]} train | {self.X_test.shape[0]} test")
        print(f"[HeartDataset] Features: {self.n_features} | Classes: {self.n_classes}")
        print(f"[HeartDataset] Équilibre — "
              f"Pas de maladie: {(y==0).sum()} | Maladie cardiaque: {(y==1).sum()}")

    def make_non_iid_partitions(
        self,
        num_clients: int = config.NUM_HOSPITALS,
        alpha: float     = config.DIRICHLET_ALPHA,
        min_samples_per_class: int = 2,  # Sécurité : 2 patients min par classe
    ) -> List[Dict]:
        """
        Partitionnement non-IID avec garantie de présence de toutes les classes.
        """
        np.random.seed(config.SEED)
        labels    = self.y_train
        n_classes = self.n_classes
        
        # 1. Identifier les indices par classe
        class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
        client_train_idx = [[] for _ in range(num_clients)]

        # 2. Distribution équilibrée du "Minimum Vital"
        for c in range(n_classes):
            idx = class_indices[c].copy()
            np.random.shuffle(idx)
            
            # Vérification de sécurité
            if len(idx) < num_clients * min_samples_per_class:
                raise ValueError(f"Pas assez de samples dans la classe {c} "
                                 f"pour garantir {min_samples_per_class} par hôpital.")

            # On donne d'abord le minimum à chaque hôpital
            for cid in range(num_clients):
                start = cid * min_samples_per_class
                end = start + min_samples_per_class
                client_train_idx[cid].extend(idx[start:end])
            
            # 3. Distribution de Dirichlet pour le RESTE des données de la classe
            remaining_idx = idx[num_clients * min_samples_per_class:]
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Calcul des parts du reste
            splits = (proportions * len(remaining_idx)).astype(int)
            splits[-1] = len(remaining_idx) - splits[:-1].sum() # Correction arrondi
            
            curr = 0
            for cid, size in enumerate(splits):
                client_train_idx[cid].extend(remaining_idx[curr:curr + size])
                curr += size

        # 4. Construction de l'objet final (identique à ton code)
        partitions = []
        for cid, t_idx in enumerate(client_train_idx):
            t_idx = np.array(t_idx)
            np.random.shuffle(t_idx) # Shuffle final pour mélanger les seeds et le reste
            
            class_counts = {c: int((labels[t_idx] == c).sum()) for c in range(n_classes)}
            partitions.append({
                "hospital_id":  f"H{cid + 1}",
                "train_idx":    t_idx,
                "class_counts": class_counts,
                "n_samples":    len(t_idx),
            })

        self._print_partitions(partitions)
        return partitions

    @staticmethod
    def _print_partitions(partitions: List[Dict]):
        print(f"\n[Partitionnement] Dirichlet α={config.DIRICHLET_ALPHA} "
              f"— {len(partitions)} hôpitaux")
        for p in partitions:
            print(f"  {p['hospital_id']}: {p['n_samples']:3d} patients | "
                  f"Sains: {p['class_counts'][0]:3d} | "
                  f"Malades: {p['class_counts'][1]:3d}")