import gymnasium as gym
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import config
import wandb
from wandb.integration.sb3 import WandbCallback


# Set seed
set_random_seed(config.SEED) #Sets seeds for random number generators in Python, NumPy, and PyTorch to ensure reproducibility.

# Create env
env = gym.make(config.ENV_NAME)
env.reset(seed=config.SEED) #Resets the environment to an initial state and sets the seed for the environment's random number generator.

#evaluation environment
eval_env = gym.make(config.ENV_NAME)

# Initialize wandb
run = wandb.init(
    project=config.WANDB_PROJECT,
    name=config.WANDB_RUN_NAME,
    config={
        "env_name": config.ENV_NAME,
        "algorithm": config.ALGORITHM,
        "total_timesteps": config.TOTAL_TIMESTEPS,
        "seed": config.SEED,
    },
    sync_tensorboard=True,  # Automatically sync SB3's tensorboard metrics to Wandb
    monitor_gym=True,        # Video recording / gym metadata tracking
    save_code=True,
)

# Initialize agent
if config.ALGORITHM == "SAC":
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE,
        reward_scale=config.REWARD_SCALE, 
        verbose=1, 
        tensorboard_log=f"./logs/tensorboard/{run.id}", 
        seed=config.SEED
        )
elif config.ALGORITHM == "DDPG":
    model = DDPG(
        "MlpPolicy", 
        env, 
        learning_rate=config.LEARNING_RATE, 
        verbose=1, 
        tensorboard_log=f"./logs/tensorboard/{run.id}", 
        seed=config.SEED
        )

#EvalCallback handles evaluation of the agent during training.
# It periodically evaluates the agent's performance on a separate evaluation environment and can save the best model based on evaluation results.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{config.CHECKPOINT_DIR}/best_model",
    log_path=f"{config.CHECKPOINT_DIR}/results",
    eval_freq=config.EVAL_FREQ,           # ← Use config instead of 10000
    n_eval_episodes=config.N_EVAL_EPISODES,  # ← Use config instead of 5
    deterministic=True,
    render=False,
)
# WandbCallback tracks hyperparams, gradients, and metrics during learn()
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"{config.MODEL_DIR}/{run.id}",
    verbose=2,
)

# Actual Training
model.learn(
    total_timesteps=config.TOTAL_TIMESTEPS,
    callback=[eval_callback, wandb_callback],
    tb_log_name=f"{config.ALGORITHM}_{config.ENV_NAME}"
)

# 7. Close Wandb run
run.finish()