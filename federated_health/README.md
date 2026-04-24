# Apprentissage Fédéré — Prédiction des Maladies Cardiaques

**Dataset :** Cleveland Heart Disease (UCI Repository) — 303 patients, 13 features, classification binaire  
**Algorithme :** FedProx (régularisation proximale)  
**Hypothèse :** Le FL atteint des performances proches du centralisé tout en préservant la confidentialité totale des données patients.

---

## Architecture du projet

```
federated_health/
├── config.py              ← tous les hyperparamètres centralisés
├── heart.csv              ← dataset Cleveland Heart Disease
├── main.py                ← point d'entrée unique
├── README.md
│
├── data/
│   └── dataset.py         ← chargement, normalisation, partitionnement Dirichlet
│
├── federated/
│   ├── client.py          ← client hôpital (entraînement local, échange de paramètres)
│   ├── serveur.py         ← serveur central (agrégation FedAvg pondérée)
│   └── simulation.py      ← boucle FL complète + baselines centralisé et local
│
├── graph/
│   └── network.py         ← graphe NetworkX + métriques + interprétation
│
├── model/
│   └── network.py         ← architecture MLP + train_one_round + evaluate
│
├── visualization/
│   └── plots.py           ← 7 figures publication-ready
│
└── results/               ← PNG générés + results.json
```

---

## Installation

```bash
pip install torch scikit-learn networkx matplotlib seaborn pandas numpy flwr
```

## Lancement

```bash
cd federated_health/
python main.py
```

---

## Protocole expérimental

Trois conditions sont évaluées sur le **même ensemble de test** (61 patients, split stratifié 80/20) :

| Condition | Description | Confidentialité |
|---|---|---|
| **Local** | Chaque hôpital s'entraîne en isolation sur ses propres données | ✅ Totale |
| **Fédéré** | FedProx, 5 hôpitaux, 30 rounds, topologie étoile | ✅ Totale |
| **Centralisé** | Toutes les données agrégées — borne supérieure théorique | ❌ Aucune |

### Hypothèse

> *F1_local << F1_fédéré ≈ F1_centralisé*

L'écart entre Fédéré et Centralisé doit être **< 5%** pour valider l'hypothèse.

---

## Résultats obtenus

| Condition | Accuracy | F1 Score | Écart vs Centralisé |
|---|---|---|---|
| Local H1 (67 patients) | 0.721 | 0.720 | — |
| Local H2 (25 patients) | 0.656 | 0.648 | — |
| Local H3 (54 patients) | 0.721 | 0.716 | — |
| Local H4 (50 patients) | 0.623 | 0.589 | — |
| Local H5 (46 patients) | 0.771 | 0.770 | — |
| **Local (moyenne)** | **0.698** | **0.689** | −16.2% |
| **★ Fédéré (FedProx)** | **0.820** | **0.815** | **−3.50%** |
| ▲ Centralisé (borne sup.) | 0.852 | 0.850 | — |

**✅ Hypothèse validée : écart = 3.50% < seuil de 5%**  
**Gain vs Local moyen : +12.69%**  
**Zéro donnée patient partagée entre hôpitaux.**

---

## Choix techniques clés

### Non-IID par distribution de Dirichlet (α = 0.75)

Les hôpitaux réels n'ont pas des distributions de patients identiques.
Un service de cardiologie spécialisé (H4 : 88% malades) ne ressemble pas
à un cabinet généraliste (H3 : 24% malades).
La distribution de Dirichlet avec α = 0.75 simule cette hétérogénéité réaliste.
Plus α est faible, plus les distributions sont déséquilibrées.

### FedProx vs FedAvg

FedAvg standard échoue sous données non-IID car les mises à jour locales
dérivent trop loin du modèle global. FedProx ajoute un terme de régularisation
proximal qui contraint chaque hôpital à ne pas trop s'éloigner du modèle global :

```
L_local = L_CE + (μ/2) * ||w_local - w_global||²
```

Avec μ = 0.01 (configurable dans `config.py`).

### Architecture MLP

Un MLP est choisi plutôt qu'un modèle à base d'arbres car :
1. Ses paramètres peuvent être moyennés entre clients (propriété requise par FedAvg)
2. Compatible avec la régularisation proximale FedProx
3. 3 762 paramètres — adapté à la taille du dataset sans surapprentissage

```
Input(13) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
          → Linear(16) → ReLU
          → Linear(2)
```

---

## Analyse graph-théorique

Le système FL est modélisé comme un graphe dirigé G = (V, E) :
- **Nœuds** : serveur central + 5 hôpitaux
- **Arêtes** : flux de paramètres (broadcast et agrégation)

| Métrique | Valeur | Interprétation |
|---|---|---|
| Centralité de degré (Serveur) | 1.0000 | Hub unique — topologie étoile |
| Centralité d'intermédiarité (Serveur) | 1.0000 | 100% des chemins inter-hôpitaux passent par le serveur |
| Coefficient de clustering moyen | 0.0000 | Aucun lien direct entre hôpitaux |
| Densité du graphe | 0.3333 | 10 arêtes / 30 possibles |
| Diamètre | 2 | Max 2 sauts entre deux hôpitaux |

Le coefficient de clustering nul prouve formellement que la confidentialité
est imposée par la **structure** du graphe, pas seulement par une règle logicielle.
Satisfait le RGPD Article 25 (Privacy by Design).

---

## Conformité RGPD

| Article | Principe | Comment le FL le satisfait |
|---|---|---|
| Art. 5(1)(c) | Minimisation des données | Seuls les poids du modèle transitent |
| Art. 17 | Droit à l'effacement | Aucune donnée patient hors de l'hôpital |
| Art. 25 | Protection dès la conception | Topologie étoile = isolation structurelle |
| Art. 83 | Responsabilité | Pas de risque de violation centralisée |

---

## Paramétrage

Tous les hyperparamètres sont dans `config.py` :

| Paramètre | Défaut | Effet |
|---|---|---|
| `NUM_ROUNDS` | 30 | Nombre de rounds de communication FL |
| `NUM_HOSPITALS` | 5 | Nombre d'hôpitaux simulés |
| `LOCAL_EPOCHS` | 5 | Époques locales par round |
| `DIRICHLET_ALPHA` | 0.75 | Hétérogénéité (plus faible = plus hétérogène) |
| `PROXIMAL_MU` | 0.01 | Force de la régularisation FedProx |
| `LEARNING_RATE` | 0.001 | Learning rate initial (décroissance exponentielle) |

---

## Figures générées dans results/

| Fichier | Contenu |
|---|---|
| `01_data_distribution.png` | Distribution non-IID des patients par hôpital |
| `02_convergence_curves.png` | F1-score et loss par round (fédéré / centralisé / local) |
| `03_comparison_heatmap.png` | Heatmap toutes conditions × toutes métriques |
| `04_fl_graph.png` | Topologie étoile + centralité + tailles des datasets |
| `05_confusion_matrices.png` | Matrices de confusion comparées |
| `06_metric_evolution.png` | Évolution accuracy, F1, précision, rappel par round |
| `07_summary_dashboard.png` | Tableau de bord de synthèse |
| `results.json` | Tous les résultats numériques (reproductibilité) |

---

## Recherche personnelle et perspectives

### L'apprentissage fédéré comme révolution du domaine médical

Le principe de l'apprentissage fédéré pourrait réellement révolutionner le domaine
médical notamment concernant la prise en charge des différents clients et réussir
à proposer les meilleures prises en charge grâce à la collaboration des médecins.

Cet apprentissage permet de faire circuler un modèle d'apprentissage entre des hôpitaux,
sans qu'il soit nécessaire d'échanger les données médicales relatives aux personnes.
Ainsi les données de base vont rester dans leurs lieux d'origine ; aucune fuite,
car le principe de base est d'éviter tout échange de données :
on ne fait circuler que les paramètres de l'algorithme d'apprentissage.

### Un gain important : les maladies rares

Un des gains importants de l'apprentissage porte sur les maladies rares :
on ne peut pas vraiment faire du machine learning sur les cancers rares
car on n'a pas assez de données sans le FL.
C'est un verrou majeur que l'apprentissage fédéré est en mesure de débloquer :
en agrégeant les rares cas disponibles dans chaque hôpital sans les centraliser,
il devient possible de construire des modèles là où aucun dataset individuel
ne serait suffisant.

### Résultats concrets dans le monde réel

Depuis 2020, Intel et l'Université de Pennsylvanie ont mené la plus grande étude
d'apprentissage fédéré du secteur médical.
Avec des ensembles de données provenant de 71 institutions sur six continents,
l'étude a démontré sa capacité à améliorer de 33% la détection des tumeurs cérébrales.

On peut voir des exemples actuels de l'apprentissage fédéré dans la recherche sur le cancer :
- Modélisation des résultats de la radiothérapie
- Histopathologie et segmentation tumorale
- Diagnostic des cancers du sein, de la prostate et du poumon
- Génomique et données multimodales

### Limites identifiées

- Hétérogénéité des données et distributions non-IID
- Charge d'étiquetage et d'annotation limitée
- Confidentialité, sécurité et fuite de modèles
- Gouvernance et confiance
- Infrastructure et évolutivité
- Acculturation des hôpitaux
- Collecte et harmonisation des données

### Pistes d'améliorations

**Confidentialité différentielle** — l'ajout d'un bruit gaussien calibré sur les gradients
renforcerait la protection contre les attaques par inversion de gradients,
qui peuvent partiellement reconstruire les données d'entraînement
à partir des mises à jour du modèle.

**Modèles fédérés multimodaux** — mettre en place l'intégration de données radiologiques,
pathologiques, etc. afin de créer des modèles prédictifs performants et de survie.

**Topologies de graphe alternatives** — passer d'une topologie étoile à un graphe
pair-à-pair (anneau, grille, aléatoire) modifierait les métriques de centralité
et le comportement de convergence, ouvrant une analyse graph-théorique comparative.

**Extension vers d'autres pathologies** — est-ce que exploiter le modèle de
l'apprentissage fédéré peut vraiment améliorer les chances de survie des patients
atteints de cancer même rare ?
En poussant l'analyse sur l'apprentissage fédéré, on pourrait à long terme
trouver des solutions concrètes dans la recherche sur le cancer.

**Perspective de thèse** — s'intéresser à Flower pour une amélioration en vue
de faire une thèse. Le projet MOJAIC (spatial transcriptomics — Jean Ogier du Terrail)
constitue une référence directe dans ce sens.

---

## Références

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Li et al. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys.
- Konečný et al. (2016). *Federated Learning: Strategies for Improving Communication Efficiency*. NeurIPS Workshop.
- Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy*.
- Cleveland Heart Disease Dataset — UCI : https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- L'apprentissage fédéré, le futur de la médecine basée sur les données : https://www.mind.eu.com/health/article/lapprentissage-federe-le-futur-de-la-medecine-basee-sur-les-donnees/
- Introduction to FL with Flower : https://github.com/gehmit/AAFAAM/blob/master/Semaine1/Lundi/Flower_1_Intro_to_FL_PyTorch.ipynb
- AI Summary Hub — Federated Learning : https://emersonbraun.github.io/ai-summary-hub/fr/docs/federated-learning/

---

## Repository GitHub

[![GitHub](https://img.shields.io/badge/GitHub-Apprentissage__federe-blue?logo=github)](https://github.com/rdsrafa/Apprentissage_federe)