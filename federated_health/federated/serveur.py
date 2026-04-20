"""
Dirichlet/federated/serveur.py
─────────────────────────────────────────────────────────────────────────────
Serveur central d'agrégation pour le système FL.

Le serveur a deux responsabilités :
  1. Coordonner le processus d'entraînement (broadcast, collecte, agrégation)
  2. Maintenir le modèle global — le seul artefact qu'il détient

Ce que le serveur N'a PAS accès :
  - Données brutes des patients de n'importe quel hôpital
  - Dossiers médicaux individuels
  - Statistiques par hôpital (uniquement les poids agrégés)

Stratégie d'agrégation : FedAvg (McMahan et al., 2017) avec moyenne
pondérée par les samples. Les hôpitaux avec plus de patients contribuent
proportionnellement davantage — statistiquement justifié.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# Racine du projet = deux niveaux au-dessus de Dirichlet/federated/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
from typing import List, Tuple, Dict

from federated_health import config
from federated_health.model.network import HeartDiseaseNet


class FederatedServeur:
    """
    Serveur central d'agrégation — implémente FedAvg.

    Le serveur détient le modèle global et orchestre chaque round :
      broadcast → collecte des mises à jour → agrégation → màj modèle global

    Attributs
    ---------
    global_model : HeartDiseaseNet
        Le modèle partagé unique maintenu par le serveur.
    round_number : int
        Round courant (utilisé pour le scheduling du LR).
    """

    def __init__(self):
        self.global_model  = HeartDiseaseNet()
        self.round_number  = 0
        n_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"[Serveur] Modèle global initialisé ({n_params:,} paramètres)")

    # ── Broadcast ─────────────────────────────────────────────────────────────

    def get_global_parameters(self) -> List[np.ndarray]:
        """
        Retourne les paramètres du modèle global.
        C'est ce que le serveur broadcast à tous les clients hôpitaux.
        """
        return self.global_model.get_parameters()

    # ── Scheduling du learning rate ───────────────────────────────────────────

    def get_current_lr(self) -> float:
        """
        Décroissance exponentielle : lr = max(lr_min, lr_0 × decay^round).
        Schedulé par le serveur et communiqué aux clients à chaque round.
        """
        lr = config.LEARNING_RATE * (config.LR_DECAY ** self.round_number)
        return max(config.LR_MIN, lr)

    # ── Agrégation FedAvg ─────────────────────────────────────────────────────

    def aggregate(
        self,
        client_updates: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:
        """
        Agrégation FedAvg : moyenne pondérée des paramètres clients.

        w_global = Σ_k (n_k / N) * w_k

        où n_k = samples locaux du client k, N = total samples.
        Équivalent mathématiquement à minimiser la somme pondérée
        des risques empiriques locaux.

        Paramètres
        ----------
        client_updates : liste de tuples (paramètres, n_samples)

        Retourne
        --------
        paramètres agrégés : liste de np.ndarray (même forme que les params du modèle)
        """
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
        """Applique les paramètres agrégés au modèle global."""
        self.global_model.set_parameters(aggregated_params)

    # ── Agrégation des métriques ──────────────────────────────────────────────

    @staticmethod
    def aggregate_metrics(
        client_metrics: List[Tuple[int, Dict]],
    ) -> Dict:
        """
        Moyenne pondérée des métriques d'évaluation entre clients.
        Utilisée pour estimer la performance globale à chaque round.
        """
        total = sum(n for n, _ in client_metrics)
        keys  = ["accuracy", "precision", "recall", "f1", "loss"]
        return {
            k: sum(n * m[k] for n, m in client_metrics) / total
            for k in keys
        }