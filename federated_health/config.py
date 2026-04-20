"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration file for the Federated Learning project.

All hyperparameters are defined here so that experiments are fully
reproducible by changing a single file. In a real deployment, these
would typically be loaded from a YAML/JSON config file or CLI arguments.
─────────────────────────────────────────────────────────────────────────────
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_PATH   = "heart.csv"
TEST_SIZE   = 0.20          # 80% train / 20% test (stratified)

# ── Federated Learning ────────────────────────────────────────────────────────
NUM_HOSPITALS    = 5        # number of simulated hospital clients
NUM_ROUNDS       = 30       # number of FL communication rounds
LOCAL_EPOCHS     = 5        # local training epochs per round per hospital
BATCH_SIZE       = 16       # mini-batch size for local training
LEARNING_RATE    = 1e-3     # initial learning rate (Adam)
LR_DECAY         = 0.97     # multiplicative LR decay per round
LR_MIN           = 1e-4     # minimum learning rate floor
PROXIMAL_MU      = 0.01     # FedProx proximal term (0 = vanilla FedAvg)
FRACTION_FIT     = 1.0      # fraction of clients participating each round
WEIGHT_DECAY     = 1e-4     # L2 regularization in Adam

# ── Non-IID Partitioning ──────────────────────────────────────────────────────
# Dirichlet concentration parameter alpha:
#   alpha → 0   : extreme non-IID (each hospital sees 1 class only)
#   alpha = 0.5 : realistic hospital heterogeneity (our default)
#   alpha → inf : IID (uniform distribution across hospitals)
DIRICHLET_ALPHA  = 0.75

# ── Model Architecture ────────────────────────────────────────────────────────
HIDDEN_DIMS      = [64, 32, 16]   # MLP hidden layer sizes
DROPOUT_RATES    = [0.3, 0.2]     # dropout after first two hidden layers
N_FEATURES       = 13             # Cleveland Heart Disease dataset features
N_CLASSES        = 2              # binary: 0=no disease, 1=heart disease

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR      = "results"

# ── Visualization ─────────────────────────────────────────────────────────────
COLORS = {
    "federated":   "#2E86AB",
    "centralized": "#A23B72",
    "local":       "#F18F01",
    "server":      "#E84855",
    "hospitals":   ["#3A86FF", "#06D6A0", "#FFB703", "#FB5607", "#8338EC"],
}
