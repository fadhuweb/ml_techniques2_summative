"""
Main Entry Point — Run Best Performing Model
==============================================

Loads the best trained model (auto-detects from models/ directory)
and runs it in the Arcade (OpenGL) visualization with terminal verbose.

Usage:
    python main.py                    # Auto-detect best model
    python main.py --model ppo        # Force specific algorithm
    python main.py --random           # Random agent (no model)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import NigerianWildlifeConservationEnv
from environment.rendering import ArcadeRenderer
from environment.world_model import ZONES, ACTIONS


def load_best_model(algorithm: str = "auto"):
    """
    Load the best trained model. Priority: PPO > DQN > REINFORCE.
    Returns (model, model_name, model_type) or (None, 'random', 'random').
    """
    search_order = {
        "auto": [
            ("models/pg/best_ppo.zip", "Best PPO", "sb3"),
            ("models/dqn/best_dqn.zip", "Best DQN", "sb3"),
            ("models/pg/best_reinforce.pt", "Best REINFORCE", "reinforce"),
        ],
        "ppo": [("models/pg/best_ppo.zip", "Best PPO", "sb3")],
        "dqn": [("models/dqn/best_dqn.zip", "Best DQN", "sb3")],
        "reinforce": [("models/pg/best_reinforce.pt", "Best REINFORCE", "reinforce")],
    }
    
    for path, name, model_type in search_order.get(algorithm, search_order["auto"]):
        if os.path.exists(path):
            if model_type == "sb3":
                try:
                    from stable_baselines3 import PPO, DQN
                    if "ppo" in path.lower():
                        model = PPO.load(path)
                    else:
                        model = DQN.load(path)
                    print(f"  Loaded: {name} from {path}")
                    return model, name, "sb3"
                except Exception as e:
                    print(f"  Failed to load {path}: {e}")
            elif model_type == "reinforce":
                try:
                    import torch
                    from training.reinforce_training import REINFORCE
                    env_temp = NigerianWildlifeConservationEnv()
                    agent = REINFORCE(
                        obs_dim=env_temp.observation_space.shape[0],
                        act_dim=env_temp.action_space.n,
                    )
                    agent.load(path)
                    env_temp.close()
                    print(f"  Loaded: {name} from {path}")
                    return agent, name, "reinforce"
                except Exception as e:
                    print(f"  Failed to load {path}: {e}")
    
    print("  No trained model found — using random agent")
    return None, "Random Agent", "random"


def get_action(model, model_type: str, obs: np.ndarray, env) -> int:
    """Get action from model regardless of type."""
    if model_type == "sb3":
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    elif model_type == "reinforce":
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            probs = model.policy(obs_t)
            return torch.argmax(probs, dim=1).item()
    else:
        return int(env.action_space.sample())


def main():
    parser = argparse.ArgumentParser(description="Run best RL agent with visualization")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "ppo", "dqn", "reinforce", "random"])
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NIGERIAN WILDLIFE CONSERVATION — RL AGENT SIMULATION")
    print("=" * 60)
    
    # Load model
    if args.model == "random":
        model, model_name, model_type = None, "Random Agent", "random"
    else:
        model, model_name, model_type = load_best_model(args.model)
    
    print(f"  Agent: {model_name}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Visualization: {'Arcade (OpenGL)' if not args.no_render else 'Terminal only'}")
    print("=" * 60)
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---\n")
        
        env = NigerianWildlifeConservationEnv(
            render_mode="human" if not args.no_render else "ansi",
            seed=42 + ep,
            max_timesteps=120,
        )
        
        renderer = None
        if not args.no_render:
            renderer = ArcadeRenderer(env)
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = get_action(model, model_type, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            
            # Render
            if renderer:
                should_close = renderer.render(
                    env.zone_states, env.episode_events,
                    env.budget, env.timestep, env.cumulative_reward,
                )
                if should_close:
                    done = True
            
            # Terminal verbose
            action_name = env.get_action_name(action)
            print(f"Step {step_count:3d} | {action_name:40s} | "
                  f"R: {reward:+.2f} | Budget: {env.budget:.1f} | "
                  f"Pop: {info['mean_wildlife_pop']:.3f} | "
                  f"Hab: {info['mean_habitat_integrity']:.3f}")
            
            # Show events
            for i, ze in enumerate(info.get("events", [])):
                if ze:
                    print(f"         !! {ZONES[i].name}: {', '.join(ze)}")
        
        # Episode summary
        summary = env.get_episode_summary()
        print(f"\n{'='*60}")
        print(f"  Episode {ep+1} Complete")
        print(f"  Reason: {info.get('termination_reason', '?')}")
        print(f"  Steps: {summary['episode_length']} | "
              f"Reward: {summary['total_reward']:.2f}")
        print(f"  Final Populations:")
        for z_name, z_state in summary['final_zone_states'].items():
            status = "OK" if z_state['wildlife_pop'] > 0.3 else ("LOW" if z_state['wildlife_pop'] > 0.1 else "CRIT")
            print(f"    {z_name:25s}: pop={z_state['wildlife_pop']:.3f} "
                  f"hab={z_state['habitat_integrity']:.3f} [{status}]")
        print(f"  Actions: {summary['action_distribution']}")
        print(f"{'='*60}")
        
        if renderer:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()# Main entry point — loads best performing model and runs the Pygame simulation
# TODO: Step 5 — implement after training is complete
