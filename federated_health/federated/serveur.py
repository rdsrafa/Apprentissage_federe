import sys
import os
import numpy as np
from typing import List, Tuple, Dict
import config
from model.network import HeartDiseaseNet
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

class FederatedServeur:

    def __init__(self):
        self.global_model  = HeartDiseaseNet()
        self.round_number  = 0
        n_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"[Serveur] Modèle global initialisé ({n_params:,} paramètres)")


    def get_global_parameters(self) -> List[np.ndarray]:
        """
        Retourne les paramètres du modèle global.
        C'est ce que le serveur broadcast à tous les clients hôpitaux.
        """
        return self.global_model.get_parameters()


    def get_current_lr(self) -> float:
        """
        Décroissance exponentielle : lr = max(lr_min, lr_0 × decay^round).
        Schedulé par le serveur et communiqué aux clients à chaque round.
        """
        lr = config.LEARNING_RATE * (config.LR_DECAY ** self.round_number)
        return max(config.LR_MIN, lr)


    def aggregate(
        self,
        client_updates: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:

        total_samples = sum(n for _, n in client_updates)
        aggregated    = None

        for params, n_samples in client_updates:
            weight = n_samples / total_samples
            if aggregated is None:
                aggregated = [layer * weight for layer in params]
            else:
                aggregated = [
                    agg + layer * weight
                    for agg, layer in zip(aggregated, params)
                ]

        return aggregated

    def update_global_model(self, aggregated_params: List[np.ndarray]):
        self.global_model.set_parameters(aggregated_params)


    @staticmethod
    def aggregate_metrics(
        client_metrics: List[Tuple[int, Dict]],
    ) -> Dict:

        total = sum(n for n, _ in client_metrics)
        keys  = ["accuracy", "precision", "recall", "f1", "loss"]
        return {
            k: sum(n * m[k] for n, m in client_metrics) / total
            for k in keys
        }