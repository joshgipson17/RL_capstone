'''
Note: We utilized Weights & Balances (wandb) to store and record our data, as well as generate plots. Wandb uses an ssh key
login associated with our githibs to be able to send the data as well as our github logins to view the data ans produced graphs
Without access to this tool, you will only be able to read the prin outs from our code in the terminal window, not see any learning
curve plots. If you would like to view plots in real time, we can set up a teams call to review.

To run code without wandb (as you have to do without ssh key access to upload data) change line 66 USE_WANDB to "False"

'''

# Environment and training
ENV_NAME = "Hopper-v4"  # Hopper-v4 or Ant-v4

# Defaults that the batch training script will override
TOTAL_TIMESTEPS = 1_000_000
ALGORITHM =  "DDPG"  # "SAC" or "DDPG"

# ============================================================
# SAC Hyperparameters
# Based on Haarnoja et al. (2018)
# ============================================================

SAC_LEARNING_RATE = 3e-4
SAC_BUFFER_SIZE = 1_000_000
SAC_BATCH_SIZE = 256
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = 1

# Neural network
SAC_NET_ARCH = [256, 256]

# Paper reward scaling
SAC_REWARD_SCALE = 5


# ============================================================
# DDPG Hyperparameters
# Based on Fujimoto et al. (2018)
# ============================================================

DDPG_LEARNING_RATE = 1e-3
DDPG_BUFFER_SIZE = 1_000_000
DDPG_LEARNING_STARTS = 100
DDPG_BATCH_SIZE = 256
DDPG_GAMMA = 0.99
DDPG_TAU = 0.005
DDPG_TRAIN_FREQ = 1
DDPG_GRADIENT_STEPS = 1

# Neural network
DDPG_NET_ARCH = [400, 300]


# ============================================================
# Evaluation
# ============================================================

EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

# ============================================================
# Logging
# ============================================================

USE_WANDB = False       # Make True to recrod data to wandb. Keep false for Prof to be able to run code
WANDB_PROJECT = "rl-capstone"

# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = "./checkpoints"
MODEL_DIR = "./models"