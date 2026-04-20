"""
Dirichlet/federated/client.py
─────────────────────────────────────────────────────────────────────────────
Client hôpital dans le système d'apprentissage fédéré.

En déploiement réel, cette classe tournerait on-premise dans chaque hôpital,
dans un environnement sécurisé. Les données brutes des patients ne quittent
jamais ce processus.

Seules ces données traversent le réseau :
  SERVEUR → CLIENT : paramètres du modèle global (poids, pas données)
  CLIENT → SERVEUR : paramètres du modèle local mis à jour (poids, pas données)

Ce module est intentionnellement indépendant du serveur — un hôpital
ne doit rien savoir de la stratégie d'agrégation.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# Racine du projet = deux niveaux au-dessus de Dirichlet/federated/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from federated_health import config
from federated_health.model.network import HeartDiseaseNet, train_one_round, evaluate


class HospitalClient:
    """
    Représente un nœud hôpital dans le système FL.

    Chaque client :
      1. Reçoit les paramètres du modèle global du serveur  (set_parameters)
      2. S'entraîne localement sur ses données privées       (local_fit)
      3. Renvoie les paramètres mis à jour au serveur        (get_parameters)
      4. Peut évaluer le modèle global sur son test local    (local_evaluate)

    Attributs
    ---------
    hospital_id : str
        Identifiant unique (ex: "H1") pour les logs et le graphe.
    n_samples : int
        Nombre de samples locaux — utilisé pour l'agrégation pondérée.
    """

    def __init__(
        self,
        hospital_id:  str,
        X_train:      np.ndarray,
        y_train:      np.ndarray,
        X_test:       np.ndarray,
        y_test:       np.ndarray,
        proximal_mu:  float = config.PROXIMAL_MU,
    ):
        self.hospital_id  = hospital_id
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.proximal_mu  = proximal_mu
        self.n_samples    = len(X_train)
        self.device       = torch.device("cpu")

        # Modèle local — initialisé avec les paramètres globaux avant chaque round
        self.model = HeartDiseaseNet()

    # ── Échange de paramètres ─────────────────────────────────────────────────

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Charge les paramètres du modèle global broadcastés par le serveur.
        Simule la réception du modèle agrégé via le réseau.
        """
        self.model.set_parameters(parameters)

    def get_parameters(self) -> List[np.ndarray]:
        """
        Retourne les paramètres locaux mis à jour à envoyer au serveur.
        Seuls les poids transitent — aucune donnée patient n'est incluse.
        """
        return self.model.get_parameters()

    # ── Entraînement local ────────────────────────────────────────────────────

    def local_fit(
        self,
        learning_rate:  float,
        local_epochs:   int,
        global_params:  Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Entraînement local sur les données privées de l'hôpital.

        Le terme proximal FedProx est appliqué si proximal_mu > 0,
        empêchant les mises à jour locales de trop s'éloigner du modèle global.
        Critique pour les données non-IID où certains hôpitaux ont des
        distributions de classes très déséquilibrées.

        Paramètres
        ----------
        learning_rate : float  — LR schedulé par le serveur pour ce round
        local_epochs  : int    — nombre d'époques locales
        global_params : liste de np.ndarray, optionnel — pour FedProx

        Retourne
        --------
        (paramètres_mis_à_jour, n_samples, métriques)
        """
        # Copie figée des params globaux pour le terme proximal
        # On aligne sur les paramètres apprenables uniquement (pas les buffers BN)
        frozen_global = None
        if self.proximal_mu > 0 and global_params is not None:
            param_keys = {name for name, _ in self.model.named_parameters()}
            state_keys = list(self.model.state_dict().keys())
            frozen_global = [
                torch.tensor(global_params[i].copy()).to(self.device)
                for i, k in enumerate(state_keys)
                if k in param_keys
            ]

        loss = train_one_round(
            model=self.model,
            X=self.X_train,
            y=self.y_train,
            epochs=local_epochs,
            lr=learning_rate,
            device=self.device,
            proximal_mu=self.proximal_mu,
            global_params=frozen_global,
        )

        return self.get_parameters(), self.n_samples, {
            "hospital":   self.hospital_id,
            "train_loss": float(loss),
        }

    # ── Évaluation locale ─────────────────────────────────────────────────────

    def local_evaluate(self) -> Dict:
        """
        Évalue le modèle courant sur l'ensemble de test partagé.
        Retourne toutes les métriques de classification.
        """
        metrics = evaluate(self.model, self.X_test, self.y_test, self.device)
        return {"hospital": self.hospital_id, **metrics}