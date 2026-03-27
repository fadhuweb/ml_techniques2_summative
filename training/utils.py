import os
import csv
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Ensure output directories exist
MODELS_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "results"

for d in [MODELS_DIR, LOGS_DIR, RESULTS_DIR,
          os.path.join(MODELS_DIR, "dqn"),
          os.path.join(MODELS_DIR, "pg"),
          os.path.join(LOGS_DIR, "dqn"),
          os.path.join(LOGS_DIR, "reinforce"),
          os.path.join(LOGS_DIR, "ppo"),
          os.path.join(RESULTS_DIR, "dqn"),
          os.path.join(RESULTS_DIR, "reinforce"),
          os.path.join(RESULTS_DIR, "ppo")]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────

def make_env(seed: Optional[int] = None, difficulty: str = "normal"):
    """
    Create a fresh environment instance.
    All algorithms must use this factory for fair comparison.
    """
    from environment.custom_env import NigerianWildlifeConservationEnv
    return NigerianWildlifeConservationEnv(
        max_timesteps=120,
        initial_budget=100.0,
        monthly_budget_income=5.0,
        render_mode=None,
        seed=seed,
        difficulty=difficulty,
    )


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, n_episodes: int = 20, seed: int = 1000, deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model over n_episodes and return performance metrics.
    
    Uses fixed seeds for reproducible comparison across algorithms.
    
    Returns:
        dict with mean_reward, std_reward, mean_length, survival_rate,
        mean_final_pop, mean_final_habitat, extinction_rate
    """
    rewards = []
    lengths = []
    final_pops = []
    final_habs = []
    extinctions = 0
    termination_reasons = []
    
    for ep in range(n_episodes):
        env = make_env(seed=seed + ep)
        obs, info = env.reset()
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(total_reward)
        lengths.append(steps)
        
        # Final state metrics
        mean_pop = np.mean([s["wildlife_pop"] for s in env.zone_states])
        mean_hab = np.mean([s["habitat_integrity"] for s in env.zone_states])
        final_pops.append(mean_pop)
        final_habs.append(mean_hab)
        
        reason = info.get("termination_reason", "unknown")
        termination_reasons.append(reason)
        if "extinction" in str(reason):
            extinctions += 1
        
        env.close()
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "survival_rate": float(1.0 - extinctions / n_episodes),
        "extinction_rate": float(extinctions / n_episodes),
        "mean_final_pop": float(np.mean(final_pops)),
        "mean_final_habitat": float(np.mean(final_habs)),
        "termination_reasons": termination_reasons,
    }


def evaluate_reinforce(policy_net, n_episodes: int = 20, seed: int = 1000) -> Dict[str, float]:
    """
    Evaluate a custom REINFORCE policy network.
    Same metrics as evaluate_model but for the PyTorch implementation.
    """
    import torch
    
    rewards = []
    lengths = []
    final_pops = []
    final_habs = []
    extinctions = 0
    termination_reasons = []
    
    policy_net.eval()
    
    for ep in range(n_episodes):
        env = make_env(seed=seed + ep)
        obs, info = env.reset()
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                probs = policy_net(obs_tensor)
                action = torch.argmax(probs, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(total_reward)
        lengths.append(steps)
        
        mean_pop = np.mean([s["wildlife_pop"] for s in env.zone_states])
        mean_hab = np.mean([s["habitat_integrity"] for s in env.zone_states])
        final_pops.append(mean_pop)
        final_habs.append(mean_hab)
        
        reason = info.get("termination_reason", "unknown")
        termination_reasons.append(reason)
        if "extinction" in str(reason):
            extinctions += 1
        
        env.close()
    
    policy_net.train()
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "survival_rate": float(1.0 - extinctions / n_episodes),
        "extinction_rate": float(extinctions / n_episodes),
        "mean_final_pop": float(np.mean(final_pops)),
        "mean_final_habitat": float(np.mean(final_habs)),
        "termination_reasons": termination_reasons,
    }


# ─────────────────────────────────────────────────────────────
# RESULTS LOGGING
# ─────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Logs hyperparameters and results for each training run.
    Writes to CSV (for the report tables) and JSON (for plotting).
    """
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.results: List[Dict[str, Any]] = []
        self.csv_path = os.path.join(RESULTS_DIR, algorithm, f"{algorithm}_experiments.csv")
        self.json_path = os.path.join(RESULTS_DIR, algorithm, f"{algorithm}_experiments.json")
    
    def log_experiment(
        self,
        run_id: int,
        hyperparams: Dict[str, Any],
        eval_metrics: Dict[str, float],
        training_time: float,
        total_timesteps: int,
        notes: str = "",
    ):
        """Log a single experiment run."""
        entry = {
            "run_id": run_id,
            "algorithm": self.algorithm,
            "timestamp": datetime.now().isoformat(),
            "total_timesteps": total_timesteps,
            "training_time_seconds": round(training_time, 1),
            **{f"hp_{k}": v for k, v in hyperparams.items()},
            **{f"eval_{k}": v for k, v in eval_metrics.items() if k != "termination_reasons"},
            "notes": notes,
        }
        self.results.append(entry)
        
        # Write CSV incrementally
        self._write_csv()
        
        # Write JSON
        with open(self.json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print(f"\n{'─'*60}")
        print(f"  Run {run_id} Complete — {self.algorithm.upper()}")
        print(f"  Mean Reward: {eval_metrics['mean_reward']:+.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  Survival Rate: {eval_metrics['survival_rate']:.0%}")
        print(f"  Mean Final Pop: {eval_metrics['mean_final_pop']:.3f}")
        print(f"  Training Time: {training_time:.0f}s")
        print(f"  Hyperparams: {hyperparams}")
        print(f"{'─'*60}\n")
    
    def _write_csv(self):
        """Write all results to CSV."""
        if not self.results:
            return
        
        keys = self.results[0].keys()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
    
    def get_best_run(self) -> Optional[Dict]:
        """Return the run with the highest mean reward."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.get("eval_mean_reward", -9999))
    
    def print_summary_table(self):
        """Print a formatted summary table of all runs."""
        if not self.results:
            print("No results yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"  {self.algorithm.upper()} — HYPERPARAMETER EXPERIMENT SUMMARY ({len(self.results)} runs)")
        print(f"{'='*80}")
        
        # Find hyperparameter columns
        hp_cols = [k for k in self.results[0].keys() if k.startswith("hp_")]
        
        header = f"{'Run':>4} | {'Mean Reward':>12} | {'Std':>8} | {'Survival':>9} | {'Pop':>6} | {'Time':>6}"
        print(header)
        print(f"{'-'*60}")
        
        for r in sorted(self.results, key=lambda x: x.get("eval_mean_reward", -9999), reverse=True):
            print(
                f"{r['run_id']:>4} | "
                f"{r.get('eval_mean_reward', 0):>+12.2f} | "
                f"{r.get('eval_std_reward', 0):>8.2f} | "
                f"{r.get('eval_survival_rate', 0):>8.0%} | "
                f"{r.get('eval_mean_final_pop', 0):>6.3f} | "
                f"{r.get('training_time_seconds', 0):>5.0f}s"
            )
        
        best = self.get_best_run()
        if best:
            print(f"\nBest: Run {best['run_id']} with reward {best.get('eval_mean_reward', 0):+.2f}")
        print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────
# SB3 TRAINING CALLBACK
# ─────────────────────────────────────────────────────────────

from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """
    Custom SB3 callback that logs training metrics at regular intervals.
    Records episode rewards, lengths, and other metrics for plotting.
    """
    
    def __init__(self, eval_freq: int = 10000, n_eval_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_results: List[Dict] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Track episode stats
        self._current_episode_reward += self.locals.get("rewards", [0])[0]
        self._current_episode_length += 1
        
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            metrics = evaluate_model(self.model, n_episodes=self.n_eval_episodes, seed=9999)
            metrics["timestep"] = self.num_timesteps
            self.eval_results.append(metrics)
            
            if self.verbose:
                print(f"  [{self.num_timesteps:>7d}] eval_reward={metrics['mean_reward']:+.2f} "
                      f"survival={metrics['survival_rate']:.0%} "
                      f"pop={metrics['mean_final_pop']:.3f}")
        
        return True
    
    def get_training_curves(self) -> Dict[str, List]:
        """Return training data for plotting."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_results": self.eval_results,
        }