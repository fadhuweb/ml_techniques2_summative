import json
import os
import sys
import numpy as np
from flask import Flask, request, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import NigerianWildlifeConservationEnv
from environment.world_model import (
    ZONES, NUM_ZONES, ACTIONS, NUM_ACTIONS, ACTION_COSTS,
    ACTION_DEFINITIONS, ZONE_CONSERVATION_PRIORITY,
)

app = Flask(__name__)


# GLOBAL STATE
# Model and environment are loaded once at startup
_model = None
_model_name = "unknown"
_env = None


def load_model():
    """
    Load the best trained model. Tries multiple paths in order of priority.
    Falls back to random policy if no model is found.
    """
    global _model, _model_name
    
    model_paths = [
        ("models/pg/best_ppo.zip", "PPO"),
        ("models/pg/best_reinforce.zip", "REINFORCE"),
        ("models/dqn/best_dqn.zip", "DQN"),
        ("models/pg/ppo_final.zip", "PPO"),
        ("models/dqn/dqn_final.zip", "DQN"),
    ]
    
    for path, name in model_paths:
        if os.path.exists(path):
            try:
                from stable_baselines3 import DQN, PPO
                if "dqn" in path.lower():
                    _model = DQN.load(path)
                else:
                    _model = PPO.load(path)
                _model_name = name
                print(f"Loaded model: {name} from {path}")
                return True
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    print("No trained model found. API will use random policy as fallback.")
    _model = None
    _model_name = "random_policy"
    return False


def get_env():
    """Get or create the environment instance."""
    global _env
    if _env is None:
        _env = NigerianWildlifeConservationEnv(seed=42)
        _env.reset()
    return _env



# API ENDPOINTS
@app.route("/health", methods=["GET"])
def health_check():
    """API health check."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_name": _model_name,
        "environment": "NigerianWildlifeConservation-v0",
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and environment configuration."""
    env = get_env()
    return jsonify({
        "model_name": _model_name,
        "model_loaded": _model is not None,
        "environment": {
            "name": "NigerianWildlifeConservation-v0",
            "observation_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.n,
            "max_timesteps": env.max_timesteps,
            "num_zones": NUM_ZONES,
            "num_actions": NUM_ACTIONS,
            "initial_budget": env.initial_budget,
        },
        "observation_features": {
            "per_zone": [
                "temperature", "rainfall", "vegetation_index",
                "wildlife_pop", "poaching_threat", "habitat_integrity",
                "last_action", "months_since_action", "has_active_event",
            ],
            "global": [
                "budget_ratio", "time_progress", "active_events",
                "mean_pop_trend", "season",
            ],
            "total_dim": 59,
        },
    })


@app.route("/zones", methods=["GET"])
def get_zones():
    """Return all zone definitions with ecological data."""
    zones_data = []
    for z in ZONES:
        zones_data.append({
            "zone_id": z.zone_id,
            "name": z.name,
            "ecosystem_type": z.ecosystem_type,
            "area_km2": z.area_km2,
            "latitude": z.latitude,
            "longitude": z.longitude,
            "key_species": z.key_species,
            "conservation_priority": ZONE_CONSERVATION_PRIORITY.get(z.zone_id, 1.0),
            "climate_vulnerability": z.climate_vulnerability,
            "baseline": {
                "temperature": z.base_temperature,
                "rainfall": z.base_rainfall,
                "vegetation_index": z.base_vegetation_index,
                "wildlife_pop": z.base_wildlife_pop,
                "poaching_threat": z.base_poaching_threat,
                "habitat_integrity": z.base_habitat_integrity,
            },
        })
    return jsonify({"zones": zones_data})


@app.route("/actions", methods=["GET"])
def get_actions():
    """Return all available actions with descriptions and costs."""
    actions_data = []
    for a in ACTION_DEFINITIONS:
        actions_data.append({
            "id": a.id,
            "name": a.name,
            "display_name": a.display_name,
            "description": a.description,
            "cost": a.cost,
            "precondition": a.precondition,
            "has_cooldown": a.cooldown_bonus,
            "primary_effects": a.primary_effects,
            "ecosystem_affinity": a.ecosystem_affinity,
        })
    return jsonify({"actions": actions_data, "total": NUM_ACTIONS})


@app.route("/predict", methods=["POST"])
def predict_action():
    data = request.get_json()
    
    if not data or "observation" not in data:
        return jsonify({"error": "Missing 'observation' field in request body"}), 400
    
    obs = np.array(data["observation"], dtype=np.float32)
    
    if obs.shape[0] != 59:
        return jsonify({
            "error": f"Observation must be 59-dimensional, got {obs.shape[0]}"
        }), 400
    
    # Get action from model (or random if no model loaded)
    if _model is not None:
        action, _states = _model.predict(obs, deterministic=True)
        action = int(action)
        
        # Get action probabilities if available (policy gradient models)
        confidence = None
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = _model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()[0]
                confidence = float(probs[action])
        except Exception:
            confidence = None
    else:
        env = get_env()
        action = int(env.action_space.sample())
        confidence = 1.0 / (NUM_ZONES * NUM_ACTIONS)
    
    # Decode action
    zone_idx = action // NUM_ACTIONS
    action_type = action % NUM_ACTIONS
    action_def = ACTION_DEFINITIONS[action_type]
    
    response = {
        "action": action,
        "action_name": action_def.name,
        "action_display_name": action_def.display_name,
        "action_type": action_type,
        "target_zone": ZONES[zone_idx].name,
        "target_zone_id": zone_idx,
        "cost": action_def.cost,
        "description": action_def.description,
        "model_used": _model_name,
    }
    
    if confidence is not None:
        response["confidence"] = round(confidence, 4)
    
    return jsonify(response)


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json() or {}
    
    steps = min(int(data.get("steps", 10)), 120)
    seed = data.get("seed", None)
    use_model = data.get("use_model", True) and _model is not None
    
    env = NigerianWildlifeConservationEnv(seed=seed)
    obs, info = env.reset()
    
    trajectory = []
    
    for step in range(steps):
        if use_model:
            action, _ = _model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(env.action_space.sample())
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        zone_idx = action // NUM_ACTIONS
        action_type = action % NUM_ACTIONS
        
        trajectory.append({
            "step": step + 1,
            "action": action,
            "action_name": ACTIONS[action_type],
            "target_zone": ZONES[zone_idx].name,
            "reward": round(reward, 4),
            "cumulative_reward": round(env.cumulative_reward, 4),
            "budget": round(env.budget, 2),
            "zone_states": {
                ZONES[i].name: {
                    "wildlife_pop": round(s["wildlife_pop"], 4),
                    "habitat_integrity": round(s["habitat_integrity"], 4),
                    "vegetation_index": round(s["vegetation_index"], 4),
                    "poaching_threat": round(s["poaching_threat"], 4),
                }
                for i, s in enumerate(env.zone_states)
            },
            "events": {
                ZONES[i].name: env.episode_events[i]
                for i in range(NUM_ZONES)
                if env.episode_events[i]
            },
        })
        
        if terminated or truncated:
            break
    
    summary = env.get_episode_summary()
    env.close()
    
    return jsonify({
        "steps_completed": len(trajectory),
        "model_used": _model_name if use_model else "random",
        "trajectory": trajectory,
        "summary": {
            "total_reward": summary["total_reward"],
            "termination_reason": summary.get("termination_reason", "in_progress"),
            "final_zone_states": summary["final_zone_states"],
            "action_distribution": summary["action_distribution"],
            "total_extreme_events": summary["total_extreme_events"],
        },
    })


@app.route("/state/current", methods=["GET"])
def get_current_state():
    """Return the current environment state as JSON (for frontend polling)."""
    env = get_env()
    
    zone_data = {}
    for i, zone in enumerate(ZONES):
        s = env.zone_states[i]
        zone_data[zone.name] = {
            "zone_id": zone.zone_id,
            "ecosystem_type": zone.ecosystem_type,
            "wildlife_pop": round(s["wildlife_pop"], 4),
            "habitat_integrity": round(s["habitat_integrity"], 4),
            "vegetation_index": round(s["vegetation_index"], 4),
            "poaching_threat": round(s["poaching_threat"], 4),
            "temperature": round(s["temperature"], 1),
            "rainfall": round(s["rainfall"], 1),
            "key_species": zone.key_species,
            "latitude": zone.latitude,
            "longitude": zone.longitude,
        }
    
    return jsonify({
        "timestep": env.timestep,
        "budget": round(env.budget, 2),
        "cumulative_reward": round(env.cumulative_reward, 4),
        "observation": env._state_to_observation().tolist(),
        "zones": zone_data,
    })



# STARTUP


if __name__ == "__main__":
    print("  Nigerian Wildlife Conservation — RL Agent API")
    print("  Serving trained model as REST endpoint")
    load_model()
    
    print(f"\nModel: {_model_name}")
    print(f"\nStarting server on http://localhost:5000")
    
    
    app.run(host="0.0.0.0", port=5000, debug=False)