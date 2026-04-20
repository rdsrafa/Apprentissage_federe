"""
Dirichlet/federated/simulation.py
─────────────────────────────────────────────────────────────────────────────
Orchestre la simulation complète du système FL.

Trois conditions expérimentales :
  1. FÉDÉRÉ     : FedProx, topologie étoile (notre contribution principale)
  2. CENTRALISÉ : toutes les données agrégées — borne supérieure théorique
  3. LOCAL      : chaque hôpital s'entraîne en isolation — pire cas

Les trois conditions permettent de valider l'hypothèse centrale :
  F1_local < F1_fédéré ≈ F1_centralisé

Si l'écart fédéré/centralisé est < 5%, on conclut que le FL atteint
une performance quasi-optimale tout en préservant la confidentialité totale
des données patients.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from typing import List, Dict

from federated_health import config
from federated_health.data.dataset import HeartDataset
from federated_health.model.network import HeartDiseaseNet, train_one_round, evaluate
from federated_health.federated.client import HospitalClient
from federated_health.federated.serveur import FederatedServeur


# ── Apprentissage Fédéré ──────────────────────────────────────────────────────

def run_federated(
    dataset:    HeartDataset,
    partitions: List[Dict],
) -> Dict:
    """
    Simulation FL complète avec agrégation FedProx.

    Chaque round implémente le protocole FL standard :
      1. Serveur broadcast les paramètres globaux à tous les hôpitaux
      2. Chaque hôpital s'entraîne localement (avec terme proximal FedProx)
      3. Les hôpitaux envoient leurs paramètres mis à jour au serveur
      4. Le serveur agrège par FedAvg pondéré
      5. Le serveur évalue le modèle global sur le test set

    Retourne
    --------
    Dict avec :
      history      : métriques par round (accuracy, f1, precision, recall, loss)
      final_metrics: métriques au dernier round
      final_eval   : dict d'évaluation complet avec prédictions (pour matrice de confusion)
      model        : modèle global final
    """
    print("\n" + "=" * 60)
    print("APPRENTISSAGE FÉDÉRÉ  (FedProx, topologie étoile)")
    print("=" * 60)

    device  = torch.device("cpu")
    serveur = FederatedServeur()

    # Instanciation d'un client par hôpital
    clients = []
    for part in partitions:
        idx = part["train_idx"]
        clients.append(HospitalClient(
            hospital_id=part["hospital_id"],
            X_train=dataset.X_train[idx],
            y_train=dataset.y_train[idx],
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            proximal_mu=config.PROXIMAL_MU,
        ))

    history = {k: [] for k in ["accuracy", "f1", "precision", "recall", "loss"]}

    for rnd in range(1, config.NUM_ROUNDS + 1):
        serveur.round_number = rnd
        lr = serveur.get_current_lr()

        # ── Étape 1 : Broadcast des paramètres globaux ────────────────────────
        global_params = serveur.get_global_parameters()

        # ── Étape 2 : Entraînement local à chaque hôpital ────────────────────
        client_updates = []
        for client in clients:
            client.set_parameters(global_params)

            updated_params, n_samples, _ = client.local_fit(
                learning_rate=lr,
                local_epochs=config.LOCAL_EPOCHS,
                global_params=global_params,
            )
            client_updates.append((updated_params, n_samples))

        # ── Étape 3 : Agrégation sur le serveur ───────────────────────────────
        aggregated_params = serveur.aggregate(client_updates)
        serveur.update_global_model(aggregated_params)

        # ── Étape 4 : Évaluation du modèle global ─────────────────────────────
        global_eval = evaluate(
            serveur.global_model, dataset.X_test, dataset.y_test, device
        )

        for k in history:
            history[k].append(global_eval[k])

        if rnd % 5 == 0 or rnd == 1:
            print(f"  Round {rnd:02d}/{config.NUM_ROUNDS} | "
                  f"Acc: {global_eval['accuracy']:.4f} | "
                  f"F1: {global_eval['f1']:.4f} | "
                  f"Loss: {global_eval['loss']:.4f} | "
                  f"lr: {lr:.5f}")

    final_metrics = {k: history[k][-1] for k in history}
    final_eval    = evaluate(
        serveur.global_model, dataset.X_test, dataset.y_test, device
    )

    print(f"\n  [Round Final {config.NUM_ROUNDS}]")
    for k in ["accuracy", "f1", "precision", "recall"]:
        print(f"  {k.capitalize():10s}: {final_metrics[k]:.4f}")

    return {
        "history":       history,
        "final_metrics": final_metrics,
        "final_eval":    final_eval,
        "model":         serveur.global_model,
    }


# ── Baseline Centralisé ───────────────────────────────────────────────────────

def run_centralized(dataset: HeartDataset) -> Dict:
    """
    Entraînement centralisé sur toutes les données combinées.

    Représente le plafond théorique de performance — ce qu'on obtiendrait
    si tous les hôpitaux partageaient leurs données brutes.
    NON conforme RGPD (viole l'article 5) — sert uniquement de comparaison.

    Budget d'entraînement équivalent au fédéré :
        NUM_ROUNDS × LOCAL_EPOCHS époques au total.
    """
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT CENTRALISÉ  (borne supérieure — non conforme RGPD)")
    print("=" * 60)

    device = torch.device("cpu")
    model  = HeartDiseaseNet()

    total_epochs = config.NUM_ROUNDS * config.LOCAL_EPOCHS
    for ep in range(total_epochs):
        lr = max(config.LR_MIN,
                 config.LEARNING_RATE * (config.LR_DECAY ** (ep // config.LOCAL_EPOCHS)))
        train_one_round(model, dataset.X_train, dataset.y_train,
                        epochs=1, lr=lr, device=device)

    metrics = evaluate(model, dataset.X_test, dataset.y_test, device)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Précision: {metrics['precision']:.4f}")
    print(f"  Rappel:    {metrics['recall']:.4f}")

    return {"metrics": metrics, "model": model}


# ── Baselines Locaux ──────────────────────────────────────────────────────────

def run_local_baselines(
    dataset:    HeartDataset,
    partitions: List[Dict],
) -> Dict:
    """
    Entraînement local uniquement : chaque hôpital s'entraîne exclusivement
    sur ses propres données, sans aucune collaboration.

    Illustre le coût de l'isolation : les hôpitaux avec peu de données
    ou des distributions déséquilibrées (H1 : 12 patients, tous sains)
    échouent complètement.

    L'écart de performance entre local et fédéré quantifie la valeur
    ajoutée de la collaboration fédérée.
    """
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT LOCAL  (baseline isolation — aucune collaboration)")
    print("=" * 60)

    device  = torch.device("cpu")
    results = {}

    for part in partitions:
        hid   = part["hospital_id"]
        idx   = part["train_idx"]
        X_loc = dataset.X_train[idx]
        y_loc = dataset.y_train[idx]

        model = HeartDiseaseNet()
        total_epochs = config.NUM_ROUNDS * config.LOCAL_EPOCHS

        for ep in range(total_epochs):
            lr = max(config.LR_MIN,
                     config.LEARNING_RATE * (config.LR_DECAY ** (ep // config.LOCAL_EPOCHS)))
            train_one_round(model, X_loc, y_loc, epochs=1, lr=lr, device=device)

        metrics = evaluate(model, dataset.X_test, dataset.y_test, device)
        results[hid] = {
            "metrics":      metrics,
            "n_samples":    part["n_samples"],
            "class_counts": part["class_counts"],
        }

        ratio_hd = part["class_counts"][1] / max(part["n_samples"], 1)
        print(f"  {hid} ({part['n_samples']:3d} patients, "
              f"{ratio_hd:.0%} malades) → "
              f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    return results