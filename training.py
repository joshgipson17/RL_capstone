import gymnasium as gym
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import config
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import sys

'''
To run the training file in Terminal run the following:
    python training.py [ALGORITH] [TIMESTEPS] [SEED]
    python training.py DDPG 10000 0

    The algorithm can be DDPG or SAC

    The environment is set in the config file in line 12. Set ENV_NAME = "Hopper-v4" OR "Ant-v4"

'''


os.environ["WANDB_DISABLE_SYMLINKS"] = "true"

# Read experiment parameters from command line
ALGORITHM = sys.argv[1]
TOTAL_TIMESTEPS = int(sys.argv[2])
SEED = int(sys.argv[3])

run_name = f"{ALGORITHM}_{config.ENV_NAME}_{TOTAL_TIMESTEPS}steps_seed{SEED}"

print("==============================")
print("Starting Training")
print("Algorithm:", ALGORITHM)
print("Timesteps:", TOTAL_TIMESTEPS)
print("Seed:", SEED)
print("==============================")
 

# Set seed
set_random_seed(SEED) #Sets seeds for random number generators in Python, NumPy, and PyTorch to ensure reproducibility.

# Create env
env = gym.make(config.ENV_NAME)
env.reset(seed=SEED) #Resets the environment to an initial state and sets the seed for the environment's random number generator.

#evaluation environment
eval_env = Monitor(gym.make(config.ENV_NAME))

# Initialize wandb
if config.USE_WANDB:
    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        config={
            "env_name": config.ENV_NAME,
            "algorithm": ALGORITHM,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
        },
        sync_tensorboard=True,  # Automatically sync SB3's tensorboard metrics to Wandb
        monitor_gym=True,        # Video recording / gym metadata tracking
        save_code=True,
    )
else:
    run = None


# Initialize agent

if ALGORITHM == "SAC":
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config.SAC_LEARNING_RATE,
        buffer_size=config.SAC_BUFFER_SIZE,
        batch_size=config.SAC_BATCH_SIZE,
        gamma=config.SAC_GAMMA,
        tau=config.SAC_TAU,
        train_freq=config.SAC_TRAIN_FREQ,
        gradient_steps=config.SAC_GRADIENT_STEPS,
        policy_kwargs=dict(net_arch=config.SAC_NET_ARCH),
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        seed=SEED
    )

elif ALGORITHM == "DDPG":
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=config.DDPG_LEARNING_RATE,
        buffer_size=config.DDPG_BUFFER_SIZE,
        learning_starts=config.DDPG_LEARNING_STARTS,
        batch_size=config.DDPG_BATCH_SIZE,
        gamma=config.DDPG_GAMMA,
        tau=config.DDPG_TAU,
        train_freq=config.DDPG_TRAIN_FREQ,
        gradient_steps=config.DDPG_GRADIENT_STEPS,
        policy_kwargs=dict(net_arch=config.DDPG_NET_ARCH),
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        seed=SEED
    )

else:
    raise ValueError(
        f"Unknown algorithm '{ALGORITHM}'. "
        "Supported algorithms are: SAC, DDPG."
    )



print("Training model:", type(model).__name__)
print("Target timesteps:", TOTAL_TIMESTEPS)
print("Seed:", SEED)


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
if config.USE_WANDB:
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{config.MODEL_DIR}/{run.id}",
        verbose=2,
    )
else:
    wandb_callback = None

# Actual Training
class StopTrainingOnTimesteps(BaseCallback):
    def __init__(self, max_timesteps):
        super().__init__()
        self.max_timesteps = max_timesteps

    def _on_step(self):
        return self.num_timesteps < self.max_timesteps

stop_callback = StopTrainingOnTimesteps(TOTAL_TIMESTEPS)

# Build callback list
callbacks = [stop_callback, eval_callback]

if config.USE_WANDB:
    callbacks.append(wandb_callback)

# Actual training
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback = callbacks,
        tb_log_name=f"{ALGORITHM}_{config.ENV_NAME}_seed{SEED}"
    )

finally:
    print("Training complete!")
    print("Algorithm:", type(model).__name__)
    print("Final timesteps:", model.num_timesteps)
    print("Completed:", run_name)

    if run is not None:
        run.finish()