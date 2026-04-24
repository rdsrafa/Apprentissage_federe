import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from typing import List, Dict
import config
from data.dataset import HeartDataset
from model.network import HeartDiseaseNet, train_one_round, evaluate
from federated.client import HospitalClient
from federated.serveur import FederatedServeur



def run_federated(
    dataset:    HeartDataset,
    partitions: List[Dict],
) -> Dict:

    print("\n" + "=" * 60)
    print("APPRENTISSAGE FÉDÉRÉ  (FedProx, topologie étoile)")
    print("=" * 60)

    device  = torch.device("cpu")
    serveur = FederatedServeur()

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

        global_params = serveur.get_global_parameters()

        client_updates = []
        for client in clients:
            client.set_parameters(global_params)

            updated_params, n_samples, _ = client.local_fit(
                learning_rate=lr,
                local_epochs=config.LOCAL_EPOCHS,
                global_params=global_params,
            )
            client_updates.append((updated_params, n_samples))

        aggregated_params = serveur.aggregate(client_updates)
        serveur.update_global_model(aggregated_params)

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



def run_centralized(dataset: HeartDataset) -> Dict:

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



def run_local_baselines(
    dataset:    HeartDataset,
    partitions: List[Dict],
) -> Dict:

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