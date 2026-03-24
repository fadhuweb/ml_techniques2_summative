# Nigerian Wildlife Conservation — Reinforcement Learning Summative

## Overview
A reinforcement learning system for optimizing conservation resource allocation across 6 real Nigerian wildlife zones under stochastic climate change. The agent learns to manage interventions (anti-poaching patrols, habitat restoration, water provision, etc.) to maximize biodiversity and ecosystem health over a 10-year horizon.

## Environment
The custom Gymnasium environment simulates:
- **6 conservation zones**: Yankari, Cross River, Chad Basin, Okomu, Gashaka Gumti, Hadejia-Nguru Wetlands
- **Stochastic climate dynamics**: seasonal cycles, global warming trend, extreme events (drought, flood, wildfire, disease)
- **8 conservation actions**: patrol, restore habitat, provide water, relocate species, engage community, monitor wildlife, emergency intervention, or do nothing
- **39-dimensional observation space**: 6 features per zone + 3 global features
- **Composite reward**: biodiversity + habitat health + stability - extinction penalty - poaching

## RL Algorithms Implemented
| Algorithm | Type | Library |
|-----------|------|---------|
| DQN | Value-Based | Stable Baselines3 |
| REINFORCE | Policy Gradient | Custom PyTorch |
| PPO | Policy Gradient | Stable Baselines3 |

Each algorithm is trained with 10 hyperparameter configurations for comparison.

## Setup
```bash
git clone https://github.com/<your_name>_rl_summative.git
cd <your_name>_rl_summative
pip install -r requirements.txt
```

## Run Best Model
```bash
python main.py
```

## Project Structure
```
project_root/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py            # Custom Gymnasium environment
│   ├── world_model.py           # Ecological model, zones, climate, rewards
│   └── rendering.py             # Pygame visualization
├── training/
│   ├── __init__.py
│   ├── dqn_training.py          # DQN training with 10 hyperparameter runs
│   ├── pg_training.py           # PPO training with 10 hyperparameter runs
│   └── reinforce_training.py    # REINFORCE training with 10 hyperparameter runs
├── models/
│   ├── dqn/                     # Saved DQN model checkpoints
│   └── pg/                      # Saved PPO + REINFORCE model checkpoints
├── assets/                      # Sprites, fonts, images for Pygame GUI
├── main.py                      # Entry point — run best model with visualization
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Author
[Your Name] — ALU [Year]
# ml_techniques2_summative
