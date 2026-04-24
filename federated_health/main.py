import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Racine du projet = un niveau au-dessus de federated_health/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import numpy as np
import torch

import config
from data.dataset import HeartDataset
from federated.simulation import (
    run_federated, run_centralized, run_local_baselines
)
from graph.network import (
    build_fl_graph, compute_graph_metrics, print_graph_interpretation
)
from visualization.plots import (
    plot_data_distribution,
    plot_convergence,
    plot_comparison_heatmap,
    plot_fl_graph,
    plot_confusion_matrices,
    plot_metric_evolution,
    plot_summary_dashboard,
)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


def save_results(fed_result, central_result, local_results, graph_metrics):
    output = {
        "config": {
            "num_hospitals":   config.NUM_HOSPITALS,
            "num_rounds":      config.NUM_ROUNDS,
            "local_epochs":    config.LOCAL_EPOCHS,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
            "proximal_mu":     config.PROXIMAL_MU,
            "learning_rate":   config.LEARNING_RATE,
        },
        "centralized": {
            k: round(float(v), 4)
            for k, v in central_result["metrics"].items()
            if k not in ("preds", "labels")
        },
        "federated": {
            k: round(float(v), 4)
            for k, v in fed_result["final_metrics"].items()
        },
        "local_hospitals": {
            hid: {
                k: round(float(v), 4)
                for k, v in r["metrics"].items()
                if k not in ("preds", "labels")
            }
            for hid, r in local_results.items()
        },
        "graph": {
            k: (round(float(v), 4) if isinstance(v, float) else v)
            for k, v in graph_metrics.items()
            if not isinstance(v, dict)
        },
    }
    path = os.path.join(config.RESULTS_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Sauvegardé : {path}")


def print_final_report(fed_result, central_result, local_results):
    print("\n" + "=" * 60)
    print("RAPPORT FINAL — VALIDATION DE L'HYPOTHÈSE")
    print("=" * 60)

    local_f1s = [r["metrics"]["f1"] for r in local_results.values()]
    fed_f1    = fed_result["final_metrics"]["f1"]
    cent_f1   = central_result["metrics"]["f1"]
    gap_pct   = abs(fed_f1 - cent_f1) * 100
    gain_pct  = abs(fed_f1 - np.mean(local_f1s)) * 100

    print(f"\n  Condition               | Accuracy | F1 Score | Précision | Rappel")
    print(f"  {'─'*66}")
    for hid, r in local_results.items():
        m = r["metrics"]
        print(f"  Local {hid} ({r['n_samples']:3d} patients)   | "
              f"  {m['accuracy']:.4f} | {m['f1']:.4f}   | {m['precision']:.4f}    | {m['recall']:.4f}")
    print(f"  {'─'*66}")
    fm = fed_result["final_metrics"]
    print(f"  ★ Fédéré (FedProx)     | "
          f"  {fm['accuracy']:.4f} | {fm['f1']:.4f}   | {fm['precision']:.4f}    | {fm['recall']:.4f}")
    cm_r = central_result["metrics"]
    print(f"  ▲ Centralisé (borne)   | "
          f"  {cm_r['accuracy']:.4f} | {cm_r['f1']:.4f}   | {cm_r['precision']:.4f}    | {cm_r['recall']:.4f}")

    print(f"\n  Local (F1 moyen) :  {np.mean(local_f1s):.4f}")
    print(f"  Local (plage)    :  [{min(local_f1s):.4f} – {max(local_f1s):.4f}]")
    print(f"  Fédéré F1        :  {fed_f1:.4f}")
    print(f"  Centralisé F1    :  {cent_f1:.4f}")
    print(f"\n  Écart  (Fédéré – Centralisé) : {gap_pct:.2f}%")
    print(f"  Gain   (Fédéré – Local moy.) : +{gain_pct:.2f}%")

    print("\n  " + "─" * 58)
    if gap_pct < 5.0:
        print(f"\n Hypothèse validée car l'écart = {gap_pct:.2f}% < seuil de 5% : ")
        print(f"  Le FL atteint {gap_pct:.2f}% de moins que le centralisé")
        print(f"  tout en gagnant +{gain_pct:.2f}% vs les modèles locaux,")
        print(f"  avec ZÉRO donnée patient partagée entre hôpitaux.")
    else:
        print(f"\n  Hypothèse pas validée car l'écart = {gap_pct:.2f}% > 5% — pour améliorer :")
        print(f"     - Augmenter NUM_ROUNDS dans config.py (actuel : {config.NUM_ROUNDS})")
        print(f"     - Augmenter DIRICHLET_ALPHA (actuel : {config.DIRICHLET_ALPHA})")
        print(f"     - Réduire PROXIMAL_MU (actuel : {config.PROXIMAL_MU})")
    print("\n" + "=" * 60)


def main():
    print("  FEDERATED LEARNING — PRÉDICTION MALADIE CARDIAQUE\n")
    print("  Cleveland Dataset | PyTorch + FedProx | 5 Hôpitaux")

    print("\n[1/7] Chargement du dataset & création des partitions hôpitaux...")
    dataset    = HeartDataset()
    partitions = dataset.make_non_iid_partitions()

    print("\n[2/7] Modélisation du système fédéré comme un graphe...")
    G             = build_fl_graph(partitions)
    graph_metrics = compute_graph_metrics(G)
    print_graph_interpretation(graph_metrics)

    print("\n[3/7] Baseline centralisé (borne supérieure)...")
    central_result = run_centralized(dataset)

    print("\n[4/7] Baselines locaux (isolation, aucune collaboration)...")
    local_results = run_local_baselines(dataset, partitions)

    print("\n[5/7] Simulation fédérée (FedProx)...")
    fed_result = run_federated(dataset, partitions)

    print("\n[6/7] Génération des visualisations...")
    plot_data_distribution(partitions, dataset)
    plot_convergence(fed_result["history"], central_result, local_results)
    plot_comparison_heatmap(fed_result, central_result, local_results)
    plot_fl_graph(G, graph_metrics, partitions)
    plot_confusion_matrices(fed_result, central_result, local_results)
    plot_metric_evolution(fed_result["history"])
    plot_summary_dashboard(fed_result, central_result, local_results)

    print("\n[7/7] Sauvegarde des résultats...")
    save_results(fed_result, central_result, local_results, graph_metrics)
    print_final_report(fed_result, central_result, local_results)

    print(f"\n  Tous les outputs dans : {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()