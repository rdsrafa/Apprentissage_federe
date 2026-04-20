# Federated Learning for Heart Disease Prediction

**Dataset:** Cleveland Heart Disease (UCI Repository) — 303 patients, 13 features, binary classification  
**Framework:** PyTorch + FedProx aggregation  
**Hypothesis:** Federated Learning achieves performance close to centralized training while preserving full patient data privacy.

---

## Project Structure

```
federated_health/
├── main.py                     ← single entry point — runs everything
├── config.py                   ← all hyperparameters (edit here to tune)
├── heart.csv                   ← dataset (Cleveland Heart Disease)
│
├── data/
│   └── dataset.py              ← loading, normalization, Dirichlet partitioning
│
├── model/
│   └── network.py              ← MLP architecture + train_one_round + evaluate
│
├── federated/
│   ├── client.py               ← HospitalClient (local training, parameter exchange)
│   ├── server.py               ← FederatedServer (FedAvg aggregation)
│   └── simulation.py           ← run_federated / run_centralized / run_local_baselines
│
├── graph/
│   └── network.py              ← NetworkX graph + metrics + interpretation
│
├── visualization/
│   └── plots.py                ← all 7 figures (publication-ready)
│
└── results/                    ← generated outputs (PNG + JSON)
```

---

## Installation

```bash
pip install torch scikit-learn networkx matplotlib seaborn pandas numpy flwr
```

## Running the Project

```bash
python3 main.py
```

This runs all 7 steps sequentially and produces all outputs in `results/`.

---

## Experimental Design

Three conditions are evaluated on the **same held-out test set** (61 patients, stratified):

| Condition | Description | Privacy |
|---|---|---|
| **Local** | Each hospital trains in isolation on its own data | ✅ Full |
| **Federated** | FedProx with 5 hospitals, 25 rounds, star topology | ✅ Full |
| **Centralized** | All data pooled together (theoretical upper bound) | ❌ None |

### Hypothesis

> *Federated F1 ≈ Centralized F1, while Local F1 << Federated F1*

The gap between Federated and Centralized must be **< 5%** to validate the hypothesis.

---

## Key Design Choices

### Non-IID Partitioning (Dirichlet, α = 0.5)
Hospitals do not have uniform class distributions. A cardiologist-focused hospital sees more heart disease cases; a general practice hospital sees fewer. The Dirichlet distribution with α = 0.5 simulates this realistic heterogeneity. Lower α → more extreme imbalance.

### FedProx vs FedAvg
Vanilla FedAvg (McMahan et al., 2017) fails under non-IID data because local updates drift away from the global optimum. FedProx (Li et al., 2020) adds a proximal regularization term:

```
L_local = L_CE + (μ/2) * ||w_local - w_global||²
```

This constrains each hospital's update to stay close to the global model (μ = 0.01 by default).

### Model Architecture
A compact MLP is used instead of tree-based models (XGBoost, etc.) because:
1. Neural network parameters can be averaged across clients (required for FedAvg)
2. Compatible with gradient-based FedProx regularization
3. BatchNorm stabilizes training under heterogeneous mini-batches

```
Input(13) → Linear(64) → BN → ReLU → Dropout(0.3)
          → Linear(32) → BN → ReLU → Dropout(0.2)
          → Linear(16) → ReLU
          → Linear(2)
```

### Graph Representation
The FL system is modeled as a directed star graph (NetworkX):
- **Server degree centrality = 1.0**: server is the unique hub
- **Clustering coefficient ≈ 0**: hospitals are never directly connected
- **Graph diameter = 2**: any two hospitals are 2 hops apart

These metrics formally demonstrate that the architecture enforces **privacy by design** (GDPR Article 25): there is no communication path between any two hospitals.

---

## Results (Cleveland Dataset)

| Condition | Accuracy | F1 Score | Gap vs Centralized |
|---|---|---|---|
| Local H1 (12 patients) | ~0.46 | ~0.29 | — |
| Local H4 (8 patients)  | ~0.54 | ~0.38 | — |
| Local average | ~0.62 | ~0.59 | -22% |
| **Federated (FedProx)** | **~0.82** | **~0.82** | **< 1%** |
| Centralized (bound) | ~0.82 | ~0.82 | 0% |

**Hypothesis validated**: Federated learning achieves < 1% gap versus centralized training with zero patient data sharing.

---

## GDPR Compliance Arguments

| GDPR Article | Principle | How FL satisfies it |
|---|---|---|
| Art. 5(1)(c) | Data minimization | Only model weights travel the network, not patient records |
| Art. 25 | Privacy by design | Star graph architecture structurally prevents inter-hospital data access |
| Art. 17 | Right to erasure | No patient data stored outside the originating hospital |
| Art. 83 | Liability | No centralized breach risk — data never aggregated |

---

## Hyperparameter Tuning

All parameters are in `config.py`. Key levers:

| Parameter | Default | Effect |
|---|---|---|
| `NUM_ROUNDS` | 30 | More rounds → better convergence (diminishing returns after ~20) |
| `DIRICHLET_ALPHA` | 0.75 | Lower → more non-IID → harder for federated |
| `PROXIMAL_MU` | 0.01 | Higher → more conservative local updates |
| `LOCAL_EPOCHS` | 5 | Higher → more computation per round, more drift risk |
| `NUM_HOSPITALS` | 5 | Simulates scale; graph metrics change with n |

---

## References

- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS. (FedAvg)
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys. (FedProx)
- Konečný, J. et al. (2016). *Federated Learning: Strategies for Improving Communication Efficiency*. NeurIPS Workshop.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. (privacy extensions)
- Cleveland Heart Disease Dataset: UCI Machine Learning Repository.
