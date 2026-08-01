# Environment and training
ENV_NAME = "Hopper-v4"  # or ant
TOTAL_TIMESTEPS = 1_000_000
ALGORITHM = "SAC"  # or "DDPG"

# Hyperparameters
LEARNING_RATE = 3e-4 
REWARD_SCALE = 1.0   

# Seeds and eval
SEED = 0
EVAL_FREQ = 10_000    
N_EVAL_EPISODES = 10

# Logging
USE_WANDB = True
WANDB_PROJECT = "rl-capstone"
WANDB_RUN_NAME = f"{ALGORITHM}_{ENV_NAME}_seed{SEED}"

# Checkpoints
CHECKPOINT_DIR = "./checkpoints"
MODEL_DIR = "./models"