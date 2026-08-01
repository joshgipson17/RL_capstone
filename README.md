# RL Capstone: SAC vs DDPG Reproduction

Reproducing the Soft Actor-Critic (SAC) paper using Stable-Baselines3.

## Setup
```bash
python -m venv sac_env
sac_env\Scripts\activate
pip install -r requirements.txt
wandb login
```

## Running Training
```bash
python training.py
```

Edit `config.py` to change environment, algorithm, or seed.

## Results
View results on wandb: [link here]
