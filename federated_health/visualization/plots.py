import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import pandas as pd
from typing import Dict, List

import config
from data.dataset import CLASS_NAMES

plt.style.use("seaborn-v0_8-paper")
RESULTS_DIR = config.RESULTS_DIR
C = config.COLORS



def _save(name: str):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")



def plot_data_distribution(partitions: List[Dict], dataset):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hospitals  = [p["hospital_id"] for p in partitions]
    no_disease = [p["class_counts"][0] for p in partitions]
    heart_dis  = [p["class_counts"][1] for p in partitions]
    x, w = np.arange(len(hospitals)), 0.35

    ax = axes[0]
    b1 = ax.bar(x - w/2, no_disease, w, label="Pas de maladie (0)",
                color="#3A86FF", edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w/2, heart_dis,  w, label="Maladie cardiaque (1)",
                color="#E84855", edgecolor="white", linewidth=1.2)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                str(int(bar.get_height())), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(hospitals, fontsize=12)
    ax.set_ylabel("Nombre de patients", fontsize=12)
    ax.set_title(f"Distribution non-IID (Dirichlet α={config.DIRICHLET_ALPHA})\n"
                 f"Simule la spécialisation réelle des hôpitaux",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    wedges, texts, autotexts = axes[1].pie(
        [p["n_samples"] for p in partitions],
        labels=hospitals, autopct="%1.1f%%",
        colors=C["hospitals"][:len(hospitals)],
        startangle=90, pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
    axes[1].set_title("Volume patients par hôpital\n"
                      "(hétérogène — reflète l'inégalité réelle)",
                      fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save("01_data_distribution.png")



def plot_convergence(fed_history: Dict, central_result: Dict, local_results: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    rounds    = list(range(1, config.NUM_ROUNDS + 1))
    local_f1s = [r["metrics"]["f1"] for r in local_results.values()]

    ax = axes[0]
    ax.plot(rounds, fed_history["f1"],
            color=C["federated"], linewidth=2.5,
            marker="o", markersize=3.5, label="Fédéré (FedProx)")

    cent_f1 = central_result["metrics"]["f1"]
    ax.axhline(cent_f1, color=C["centralized"], linewidth=2.2, linestyle="--",
               label=f"Centralisé (F1 = {cent_f1:.3f})  ← borne supérieure")

    avg_local = np.mean(local_f1s)
    ax.axhline(avg_local, color=C["local"], linewidth=2.2, linestyle=":",
               label=f"Local moyen (F1 = {avg_local:.3f})")

    ax.fill_between(rounds, min(local_f1s), max(local_f1s),
                    alpha=0.12, color=C["local"],
                    label=f"Plage locale [{min(local_f1s):.2f} – {max(local_f1s):.2f}]")

    gap = abs(fed_history["f1"][-1] - cent_f1)
    ax.annotate(f"Écart = {gap*100:.1f}%",
                xy=(rounds[-1], fed_history["f1"][-1]),
                xytext=(rounds[-1] - 9, fed_history["f1"][-1] - 0.07),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=10, color="gray")

    ax.set_xlabel("Round de communication", fontsize=12)
    ax.set_ylabel("F1 Score pondéré", fontsize=12)
    ax.set_title("Fédéré vs Centralisé vs Local — F1 Score\n"
                 "Résultat clé : Fédéré ≈ Centralisé  |  Local << Fédéré",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, fed_history["loss"],
            color=C["federated"], linewidth=2.5,
            marker="s", markersize=3.5, label="Loss validation fédérée")
    ax.set_xlabel("Round de communication", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Fédéré — Convergence de la loss\n"
                 f"LR : {config.LEARNING_RATE} → {config.LR_MIN} (décroissance {config.LR_DECAY}/round)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save("02_convergence_curves.png")



def plot_comparison_heatmap(fed_result: Dict, central_result: Dict, local_results: Dict):
    rows, accs, precs, recs, f1s = [], [], [], [], []

    for hid, r in local_results.items():
        rows.append(f"Local {hid}  ({r['n_samples']} pts)")
        accs.append(r["metrics"]["accuracy"])
        precs.append(r["metrics"]["precision"])
        recs.append(r["metrics"]["recall"])
        f1s.append(r["metrics"]["f1"])

    rows.append("Fédéré (FedProx)")
    accs.append(fed_result["final_metrics"]["accuracy"])
    precs.append(fed_result["final_metrics"]["precision"])
    recs.append(fed_result["final_metrics"]["recall"])
    f1s.append(fed_result["final_metrics"]["f1"])

    rows.append("Centralisé (borne sup.)")
    accs.append(central_result["metrics"]["accuracy"])
    precs.append(central_result["metrics"]["precision"])
    recs.append(central_result["metrics"]["recall"])
    f1s.append(central_result["metrics"]["f1"])

    df = pd.DataFrame(
        {"Accuracy": accs, "Précision": precs, "Rappel": recs, "F1 Score": f1s},
        index=rows,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.8, ax=ax, vmin=0.3, vmax=1.0,
                cbar_kws={"label": "Score", "shrink": 0.8},
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Comparaison de performance — Local vs Fédéré vs Centralisé\n"
                 "Cleveland Heart Disease Dataset (61 patients de test)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=11)

    n_local = len(local_results)
    for row_idx, color in [(n_local, C["federated"]), (n_local + 1, C["centralized"])]:
        ax.add_patch(plt.Rectangle(
            (0, row_idx), df.shape[1], 1,
            fill=False, edgecolor=color, lw=3, clip_on=False,
        ))

    plt.tight_layout()
    _save("03_comparison_heatmap.png")



def plot_fl_graph(G: nx.DiGraph, metrics: Dict, partitions: List[Dict]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hospitals = [n for n in G.nodes if G.nodes[n]["type"] == "hospital"]

    ax = axes[0]
    pos = {"Serveur": np.array([0, 0])}
    for i, h in enumerate(hospitals):
        angle = 2 * np.pi * i / len(hospitals)
        pos[h] = np.array([1.6 * np.cos(angle), 1.6 * np.sin(angle)])

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, arrows=True,
                           arrowsize=22, edge_color="#888888", width=1.8,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, nodelist=["Serveur"],
                           node_color=C["server"], node_size=2800, ax=ax, alpha=0.95)
    for i, h in enumerate(hospitals):
        nx.draw_networkx_nodes(G, pos, nodelist=[h],
                               node_color=C["hospitals"][i], node_size=1300,
                               ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                            font_weight="bold", font_size=12)
    ax.set_title("Graphe FL — Topologie étoile\nParamètres uniquement, zéro donnée patient",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    legend_elems = [
        mpatches.Patch(color=C["server"],       label="Serveur central"),
        mpatches.Patch(color=C["hospitals"][0],  label="Nœud hôpital"),
        mpatches.Patch(color="#888888", alpha=0.6, label="Échange de paramètres (↔)"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", fontsize=9)

    ax = axes[1]
    nodes   = list(metrics["degree_centrality"].keys())
    values  = list(metrics["degree_centrality"].values())
    bar_col = [C["server"] if n == "Serveur" else C["hospitals"][i-1]
               for i, n in enumerate(nodes)]
    bars = ax.bar(nodes, values, color=bar_col, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Centralité de degré", fontsize=11)
    ax.set_title("Centralité des nœuds\nServeur = centralité maximale",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[2]
    hids  = [p["hospital_id"] for p in partitions]
    sizes = [p["n_samples"]   for p in partitions]
    bars  = ax.barh(hids, sizes,
                    color=C["hospitals"][:len(hids)], edgecolor="white", linewidth=1.2)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(s), va="center", fontsize=10)
    ax.set_xlabel("Samples d'entraînement", fontsize=11)
    ax.set_title("Tailles des datasets hôpitaux\nPartitionnement Dirichlet non-IID",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(sizes) * 1.2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save("04_fl_graph.png")



def plot_confusion_matrices(
    fed_result: Dict, central_result: Dict, local_results: Dict,
):
    from sklearn.metrics import confusion_matrix

    sorted_local = sorted(local_results.items(), key=lambda x: x[1]["metrics"]["f1"])
    conditions = {
        f"Local {sorted_local[0][0]}\n(pire hôpital)":   sorted_local[0][1]["metrics"],
        f"Local {sorted_local[-1][0]}\n(meilleur hôpital)": sorted_local[-1][1]["metrics"],
        "★ Fédéré\n(FedProx)":         fed_result["final_eval"],
        "▲ Centralisé\n(borne sup.)":  central_result["metrics"],
    }
    title_colors = [C["local"], C["local"], C["federated"], C["centralized"]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (title, m), color in zip(axes, conditions.items(), title_colors):
        cm = confusion_matrix(m["labels"], m["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    cbar=False, linewidths=0.5, linecolor="white",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{title}\nF1 = {m['f1']:.3f}",
                     fontsize=10, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("Prédit", fontsize=10)
        ax.set_ylabel("Réel", fontsize=10)

    plt.suptitle("Matrices de confusion — Cleveland Heart Disease (61 patients de test)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save("05_confusion_matrices.png")



def plot_metric_evolution(fed_history: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rounds = list(range(1, config.NUM_ROUNDS + 1))

    configs_plot = [
        ("accuracy",  "Accuracy",   axes[0][0]),
        ("f1",        "F1 Score",   axes[0][1]),
        ("precision", "Précision",  axes[1][0]),
        ("recall",    "Rappel",     axes[1][1]),
    ]

    for metric, label, ax in configs_plot:
        vals = fed_history[metric]
        ax.plot(rounds, vals, color=C["federated"], linewidth=2.5,
                marker="o", markersize=4)
        ax.fill_between(rounds, vals, alpha=0.15, color=C["federated"])
        ax.set_xlabel("Round de communication", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Fédéré — {label} par Round", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.05)
        final = vals[-1]
        ax.annotate(f"Final : {final:.3f}",
                    xy=(rounds[-1], final),
                    xytext=(rounds[-1] - 7, final - 0.09),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=10, color=C["federated"], fontweight="bold")

    plt.suptitle("Apprentissage Fédéré — Évolution de toutes les métriques par Round",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("06_metric_evolution.png")



def plot_summary_dashboard(
    fed_result: Dict, central_result: Dict, local_results: Dict,
):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#0D1117")
    fig.text(0.5, 0.96,
             "Federated Learning — Prédiction Maladie Cardiaque | Tableau de Bord",
             ha="center", fontsize=15, fontweight="bold", color="white")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.90, bottom=0.08, left=0.06, right=0.97)

    metric_keys   = ["accuracy", "f1", "precision", "recall"]
    metric_labels = ["Accuracy", "F1 Score", "Précision", "Rappel"]

    local_vals = {k: np.mean([r["metrics"][k] for r in local_results.values()])
                  for k in metric_keys}
    fed_vals   = fed_result["final_metrics"]
    cent_vals  = central_result["metrics"]

    ax = fig.add_subplot(gs[:, :2])
    ax.set_facecolor("#161B22")
    x, w = np.arange(len(metric_labels)), 0.25

    for i, (vals, label, color) in enumerate([
        ([local_vals[k] for k in metric_keys], "Local (moy.)",  C["local"]),
        ([fed_vals[k]   for k in metric_keys], "Fédéré",        C["federated"]),
        ([cent_vals[k]  for k in metric_keys], "Centralisé",    C["centralized"]),
    ]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=color, alpha=0.9, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12, color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12, color="white")
    ax.set_title("Performance : Local vs Fédéré vs Centralisé",
                 fontsize=13, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="white")
    ax.legend(fontsize=10, loc="lower right",
              facecolor="#1E2530", edgecolor="#444", labelcolor="white")

    gap_pct  = abs(fed_vals["f1"] - cent_vals["f1"]) * 100
    gain_pct = abs(fed_vals["f1"] - local_vals["f1"]) * 100

    stat_data = [
        ("F1 Fédéré",        f'{fed_vals["f1"]:.3f}',    C["federated"]),
        ("F1 Centralisé",    f'{cent_vals["f1"]:.3f}',   C["centralized"]),
        ("Écart Fed–Cent",   f'{gap_pct:.1f}%',           "#FFB703"),
        ("Gain vs Local",    f'+{gain_pct:.1f}%',         "#06D6A0"),
        ("Hôpitaux",         str(config.NUM_HOSPITALS),   "#8338EC"),
        ("Rounds FL",        str(config.NUM_ROUNDS),      "#FF6B6B"),
    ]

    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.set_facecolor("#161B22")
    ax_stats.axis("off")
    ax_stats.set_title("Statistiques Clés", fontsize=12, fontweight="bold",
                        color="white", pad=10)

    for i, (label, value, color) in enumerate(stat_data):
        y_pos = 0.88 - i * 0.15
        ax_stats.text(0.05, y_pos, label, transform=ax_stats.transAxes,
                      fontsize=10, color="#AAAAAA")
        ax_stats.text(0.95, y_pos, value, transform=ax_stats.transAxes,
                      fontsize=14, fontweight="bold", color=color, ha="right")

    plt.savefig(os.path.join(RESULTS_DIR, "07_summary_dashboard.png"),
                dpi=200, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  Sauvegardé : {os.path.join(RESULTS_DIR, '07_summary_dashboard.png')}")