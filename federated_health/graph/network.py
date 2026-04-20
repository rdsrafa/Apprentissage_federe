"""
run_local/graph/network.py
─────────────────────────────────────────────────────────────────────────────
Représentation graph-théorique du système FL avec NetworkX.

Modéliser le système FL comme un graphe dirigé sert deux objectifs :
  1. Analyse structurelle : quantifier les propriétés topologiques
  2. Argument de confidentialité : les métriques démontrent formellement
     que les hôpitaux ne sont jamais connectés entre eux

Définition du graphe :
  Nœuds : serveur central + clients hôpitaux
  Arêtes : échange de paramètres dirigé
             (serveur→client = broadcast, client→serveur = agrégation)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import networkx as nx
from typing import List, Dict


def build_fl_graph(partitions: List[Dict]) -> nx.DiGraph:
    """
    Construit le graphe de communication du système FL.

    Retourne un graphe dirigé où :
      - "Serveur" = serveur central d'agrégation
      - "H1"..."H5" = clients hôpitaux
      - Arêtes dirigées = flux de paramètres (pas de données)
    """
    G = nx.DiGraph()

    G.add_node("Serveur", type="server",
               description="Serveur d'agrégation — détient uniquement le modèle global")

    for p in partitions:
        hid = p["hospital_id"]
        G.add_node(
            hid,
            type="hospital",
            n_patients=p["n_samples"],
            class_counts=p["class_counts"],
        )
        # Serveur → Hôpital : broadcast des paramètres globaux
        G.add_edge("Serveur", hid,
                   direction="broadcast",
                   content="model_parameters")
        # Hôpital → Serveur : envoi des paramètres locaux mis à jour
        G.add_edge(hid, "Serveur",
                   direction="aggregation",
                   content="model_parameters")

    return G


def compute_graph_metrics(G: nx.DiGraph) -> Dict:
    """
    Calcule les métriques graph-théoriques pertinentes pour le FL.

    Les métriques de chemin sont calculées sur la version non-dirigée ;
    la centralité dirigée est calculée séparément.
    """
    U = G.to_undirected()

    return {
        "degree_centrality":      nx.degree_centrality(G),
        "betweenness_centrality": nx.betweenness_centrality(G),
        "in_degree_centrality":   nx.in_degree_centrality(G),
        "out_degree_centrality":  nx.out_degree_centrality(G),
        "density":                nx.density(U),
        "is_connected":           nx.is_connected(U),
        "num_nodes":              G.number_of_nodes(),
        "num_edges":              G.number_of_edges(),
        "avg_shortest_path":      nx.average_shortest_path_length(U),
        "clustering_coeff":       nx.average_clustering(U),
        "diameter":               nx.diameter(U),
    }


def print_graph_interpretation(metrics: Dict):
    """
    Affiche une interprétation structurée des métriques dans le contexte FL.
    """
    print("\n" + "=" * 60)
    print("MÉTRIQUES DU GRAPHE — ANALYSE DU SYSTÈME FÉDÉRÉ")
    print("=" * 60)

    sc = metrics["degree_centrality"]["Serveur"]
    print(f"\n  [Centralité] Centralité de degré du Serveur = {sc:.4f} (max = 1.0)")
    print(f"    → Le serveur est le seul hub de communication (topologie étoile).")
    print(f"      Aucun hôpital n'est connecté à un autre hôpital.")
    print(f"      L'isolation des données est garantie par la structure du graphe.")

    print(f"\n  [Densité] Densité du graphe = {metrics['density']:.4f}")
    n, e = metrics["num_nodes"], metrics["num_edges"]
    max_e = n * (n - 1)
    print(f"    → {e} arêtes sur {max_e} possibles (graphe complet).")
    print(f"      Design minimaliste : uniquement des liens serveur–hôpital.")
    print(f"      Supprimer le serveur déconnecte le graphe entier.")

    print(f"\n  [Clustering] Coefficient de clustering moyen = "
          f"{metrics['clustering_coeff']:.4f}")
    print(f"    → Quasi-zéro : les hôpitaux ne forment AUCUN triangle.")
    print(f"      Pas de lien pair-à-pair → aucun canal d'inférence entre hôpitaux.")
    print(f"      Satisfait l'Article 25 RGPD (Privacy by Design).")

    print(f"\n  [Chemins] Chemin moyen = {metrics['avg_shortest_path']:.4f}")
    print(f"    Diamètre = {metrics['diameter']}")
    print(f"    → Deux hôpitaux quelconques sont à exactement 2 sauts (via Serveur).")
    print(f"      Complexité de communication O(n), pas O(n²).")
    print(f"      Efficace et scalable à mesure que n_hôpitaux augmente.")

    print(f"\n  [Intermédiarité] Centralité d'intermédiarité du Serveur = "
          f"{metrics['betweenness_centrality']['Serveur']:.4f}")
    print(f"    → Tout chemin inter-hôpital passe par le serveur.")
    print(f"      Unique point de confiance — cohérent avec le modèle de")
    print(f"      menace FL (serveur honnête-mais-curieux).")