import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Parameters (from Table 3) ---
# Network and Node Parameters
N_NODES = 20
CPU_CAPACITY_RANGE = [2.0, 8.0]  # GHz
STORAGE_CAPACITY_RANGE = [10.0, 40.0] # GB
BANDWIDTH_RANGE = [100.0, 1000.0] # Mbps
INITIAL_LAYER_PRESENCE = 0.1 # 10% of layers are pre-cached

# Service and Container Parameters
N_SERVICES = 100
N_UNIQUE_LAYERS = 500
LAYERS_PER_SERVICE_RANGE = [5, 15]
LAYER_SIZE_RANGE = [10, 500] # MB
SERVICE_WORKLOAD_RANGE = [10, 50] # Giga-cycles
MAX_LATENCY_RANGE = [1.0, 3.0] # seconds
BASE_REVENUE = 10.0 # A fixed base revenue for simplicity

# RL and Model Parameters
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
STRATEGIC_FACTOR_LAMBDA = 0.5
TAU_SOFT_UPDATE = 0.005 # Soft update parameter for target networks

# --- Model Architecture ---
GCN_HIDDEN_LAYERS = [16, 16]
FM_LATENT_DIM = 16
DDPG_ACTOR_HIDDEN_SIZE = 128
DUELING_DQN_HIDDEN_SIZE = 128

# --- Training and Evaluation ---
N_TRAINING_EPISODES = 200
N_EVALUATION_RUNS = 20
P_FAIL_REWARD = 20.0 # Large negative penalty for invalid actions

# Normalization constants for reward function
# These should be tuned to make latency and storage costs comparable to revenue
C1_LATENCY_COST_NORMALIZER = 5.0
C2_STORAGE_COST_NORMALIZER = 0.01