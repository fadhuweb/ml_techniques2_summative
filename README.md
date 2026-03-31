# Nigerian Wildlife Conservation — Reinforcement Learning Summative

## Overview
A reinforcement learning system for optimizing conservation resource allocation across 6 real Nigerian wildlife zones under stochastic climate change. The agent learns to manage interventions (anti-poaching patrols, habitat restoration, water provision, species relocation, community engagement, wildlife monitoring, emergency response) to maximize biodiversity and ecosystem health over a 10-year (120-month) horizon.

## Environment
The custom Gymnasium environment simulates:
- **6 conservation zones**: Yankari, Cross River, Chad Basin, Okomu, Gashaka Gumti, Hadejia-Nguru Wetlands
- **Stochastic climate dynamics**: seasonal cycles, global warming trend, extreme events (drought, flood, wildfire, disease)
- **8 conservation actions** with ecosystem-specific effectiveness multipliers and cooldown mechanics
- **59-dimensional observation space**: 9 features per zone x 6 zones + 5 global features
- **Discrete(48) action space**: 6 zones x 8 interventions
- **9-component composite reward** with IUCN priority weighting

## RL Algorithms Implemented
| Algorithm | Type | Library | Best Reward | Survival |
|-----------|------|---------|-------------|----------|
| DQN | Value-Based | Stable Baselines3 | 320.2 | 100% |
| REINFORCE | Policy Gradient | Custom PyTorch | 8.8 | 0% |
| PPO | Policy Gradient | Stable Baselines3 | 359.1 | 100% |

Each algorithm is trained for 500,000 timesteps with 10 hyperparameter configurations (30 total experiments).

## Setup
```bash
git clone https://github.com/fadhuweb/ml_techniques2_summative.git
cd ml_techniques2_summative
pip install -r requirements.txt
```

## Run Best Model
```bash
python main.py                    # Auto-detect best model + Arcade visualization
python main.py --model ppo        # Force PPO
python main.py --model dqn        # Force DQN
python main.py --random           # Random agent (no model)
python main.py --no-render        # Terminal only (no GUI)
```

## Training
```bash
python -m training.dqn_training              # All 10 DQN experiments
python -m training.reinforce_training        # All 10 REINFORCE experiments
python -m training.pg_training               # All 10 PPO experiments
python -m training.dqn_training --run 1      # Single experiment
```

## Analysis and Plots
```bash
python -m training.analysis                  # Generate all 8 report plots
python -m training.generate_tables           # Generate hyperparameter tables
```

## Visualization
Built with **Arcade** (OpenGL-based via Pyglet). Features:
- Stylized Nigeria map with color-coded zone markers
- Real-time health bars, population sparkline, budget HUD
- Pulse animations for active interventions, flash effects for extreme events
- Terminal verbose output alongside GUI

```bash
python -m environment.rendering              # Random agent demo (no model)
```

## JSON API for Production Integration

A Flask REST API (`api.py`) serves the trained PPO model as a web-accessible endpoint, demonstrating how the RL agent can be serialized to JSON and integrated into any web or mobile application.

### Start the API Server
```bash
python api.py
```
The server runs on `http://localhost:5000`.

### API Endpoints

#### 1. Health Check
Verify the API is running and which model is loaded.
```bash
curl http://localhost:5000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "PPO"
}
```

#### 2. Get Zone Information
Returns all 6 conservation zones with species, coordinates, and ecological data.
```bash
curl http://localhost:5000/zones
```

#### 3. Get Available Actions
Returns all 8 conservation interventions with descriptions, costs, and ecosystem effectiveness.
```bash
curl http://localhost:5000/actions
```

#### 4. Get Model Info
Returns model metadata, observation space dimensions, and environment configuration.
```bash
curl http://localhost:5000/model/info
```

#### 5. Get Current Environment State
Returns the live environment state as JSON (for frontend polling).
```bash
curl http://localhost:5000/state/current
```

#### 6. Predict Best Action
Send a 59-dimensional observation vector and get the model's recommended action.

**Using curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.5,0.3,0.7,0.5,0.4,0.6,0.0,0.0,0.0,0.8,0.1,0.0,0.5,0.5]}'
```

**Using Postman:**
1. Method: `POST`, URL: `http://localhost:5000/predict`
2. Body tab → select **raw** → change dropdown to **JSON**
3. Paste the JSON body above

**Response:**
```json
{
  "action": 45,
  "action_display_name": "Community Engagement",
  "action_name": "community_engagement",
  "target_zone": "Hadejia-Nguru Wetlands",
  "target_zone_id": 5,
  "cost": 0.1,
  "confidence": 0.79,
  "model_used": "PPO",
  "description": "Fund local community conservation programs..."
}
```

The 59 numbers represent the current state of all 6 zones (9 features each: temperature, rainfall, vegetation, wildlife population, poaching threat, habitat integrity, last action, months since action, active event) plus 5 global features (budget ratio, time progress, active events, population trend, season).

#### 7. Run Simulation
Run a multi-step simulation and get the full trajectory with zone states at each step.

**Using curl:**
```bash
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"steps": 10, "seed": 42, "use_model": true}'
```

**Using Postman:**
1. Method: `POST`, URL: `http://localhost:5000/simulate`
2. Body tab → select **raw** → change dropdown to **JSON**
3. Paste: `{"steps": 10, "seed": 42, "use_model": true}`

**Parameters:**
- `steps` — number of months to simulate (max 120)
- `seed` — random seed for reproducible climate events
- `use_model` — `true` for trained PPO agent, `false` for random actions

**Response:**
```json
{
  "model_used": "PPO",
  "steps_completed": 10,
  "summary": {
    "total_reward": 23.27,
    "action_distribution": {
      "community_engagement": 6,
      "anti_poaching_patrol": 3,
      "emergency_intervention": 1
    },
    "final_zone_states": { "..." }
  },
  "trajectory": [
    {
      "step": 1,
      "action_name": "community_engagement",
      "target_zone": "Hadejia-Nguru Wetlands",
      "reward": 2.22,
      "budget": 104.0,
      "zone_states": { "..." }
    }
  ]
}
```

## Project Structure
```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── world_model.py           # Ecological model, zones, climate, rewards
│   └── rendering.py             # Arcade (OpenGL) visualization GUI components
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO using SB3
│   ├── reinforce_training.py    # Training script for REINFORCE (custom PyTorch)
│   ├── utils.py                 # Shared training utilities, evaluation, logging
│   ├── analysis.py              # Plot generation for report
│   └── generate_tables.py       # Hyperparameter table generation
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved PPO + REINFORCE models
├── main.py                      # Entry point for running best performing model
├── api.py                       # Flask JSON API for production integration
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Author
[Your Name] — ALU [Year]