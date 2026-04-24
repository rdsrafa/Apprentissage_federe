import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

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



class HeartDataset:
    def __init__(self, path: str = config.DATA_PATH):

        df = pd.read_csv(path)

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["target"].values.astype(np.int64)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y,
        )
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
        min_samples_per_class: int = 2,  
    ) -> List[Dict]:
        np.random.seed(config.SEED)
        labels    = self.y_train
        n_classes = self.n_classes
        
        class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
        client_train_idx = [[] for _ in range(num_clients)]

        for c in range(n_classes):
            idx = class_indices[c].copy()
            np.random.shuffle(idx)
            
            if len(idx) < num_clients * min_samples_per_class:
                raise ValueError(f"Pas assez de samples dans la classe {c} "
                                 f"pour garantir {min_samples_per_class} par hôpital.")

            for cid in range(num_clients):
                start = cid * min_samples_per_class
                end = start + min_samples_per_class
                client_train_idx[cid].extend(idx[start:end])
                
            remaining_idx = idx[num_clients * min_samples_per_class:]
            proportions = np.random.dirichlet([alpha] * num_clients)
            splits = (proportions * len(remaining_idx)).astype(int)
            splits[-1] = len(remaining_idx) - splits[:-1].sum() # Correction arrondi
            
            curr = 0
            for cid, size in enumerate(splits):
                client_train_idx[cid].extend(remaining_idx[curr:curr + size])
                curr += size
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