# Reproductibilité — graine fixée pour avoir les mêmes résultats à chaque exécution
SEED = 42

# Dataset
DATA_PATH   = "heart.csv"
TEST_SIZE   = 0.20          # 80% entraînement / 20% test (split stratifié)

# Apprentissage Fédéré 
NUM_HOSPITALS    = 5        # Nombre d'hôpitaux simulés dans l'expérience
NUM_ROUNDS       = 30       # Nombre de rounds de communication du modèle fédéré
LOCAL_EPOCHS     = 5        # Nombre d'époques d'entraînement local par round et par hôpital
BATCH_SIZE       = 16       # Taille des mini-batchs pour l'entraînement local
LEARNING_RATE    = 1e-3     # Learning rate initial (optimiseur Adam)
LR_DECAY         = 0.97     # Décroissance multiplicative du learning rate à chaque round
LR_MIN           = 1e-4     # Valeur plancher du learning rate (ne descend jamais en dessous)
PROXIMAL_MU      = 0.01     # Terme proximal FedProx — 0 = FedAvg classique, > 0 = FedProx
FRACTION_FIT     = 1.0      # Fraction des clients participant à chaque round (1.0 = tous)
WEIGHT_DECAY     = 1e-4     # Régularisation L2 dans Adam

# Partitionnement Non-IID
# Paramètre alpha de la distribution de Dirichlet :
#   alpha → 0    : non-IID extrême (chaque hôpital ne voit qu'une seule classe)
#   alpha = 0.75 : hétérogénéité réaliste entre hôpitaux (notre choix)
#   alpha → inf  : IID (distribution uniforme entre tous les hôpitaux)
DIRICHLET_ALPHA  = 0.75

# Architecture du Modèle 
HIDDEN_DIMS      = [64, 32, 16]   # Tailles des couches cachées du MLP
DROPOUT_RATES    = [0.3, 0.2]     # Dropout après les deux premières couches cachées
N_FEATURES       = 13             # Nombre de features du dataset Cleveland Heart Disease
N_CLASSES        = 2              # Binaire : 0 = pas de maladie, 1 = maladie cardiaque

# Chemins
RESULTS_DIR      = "results"

# Visualisation 
COLORS = {
    "federated":   "#2E86AB",
    "centralized": "#A23B72",
    "local":       "#F18F01",
    "server":      "#E84855",
    "hospitals":   ["#3A86FF", "#06D6A0", "#FFB703", "#FB5607", "#8338EC"],
}