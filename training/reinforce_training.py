"""
REINFORCE Training Script — Vanilla Policy Gradient
=====================================================

Custom PyTorch implementation of the REINFORCE algorithm since
Stable Baselines3 does not include vanilla REINFORCE.

REINFORCE uses Monte Carlo returns (no value baseline in the pure form)
to update a policy network via the policy gradient theorem:
    ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) * G_t]

Runs 10 hyperparameter experiments varying:
learning_rate, gamma, hidden layer sizes, entropy coefficient,
max_episodes_per_update, baseline (with/without), optimizer.

Usage:
    python -m training.reinforce_training
    python -m training.reinforce_training --run 3
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import (
    make_env, evaluate_reinforce, ExperimentLogger,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR,
)

TOTAL_TIMESTEPS = 500_000


# ─────────────────────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Simple MLP policy network for REINFORCE."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_layers: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample an action and return (action, log_prob)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate_action(self, obs_t: torch.Tensor, act_t: torch.Tensor):
        """Compute log_prob and entropy for given obs-action pairs."""
        probs = self.forward(obs_t)
        dist = Categorical(probs)
        return dist.log_prob(act_t), dist.entropy()


# ─────────────────────────────────────────────────────────────
# REINFORCE ALGORITHM
# ─────────────────────────────────────────────────────────────

class REINFORCE:
    """
    Vanilla REINFORCE with optional baseline subtraction.
    
    Pure REINFORCE:  ∇J = E[Σ_t ∇log π(a_t|s_t) * G_t]
    With baseline:   ∇J = E[Σ_t ∇log π(a_t|s_t) * (G_t - b)]
    where b is the running average of returns.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_layers: List[int] = [256, 256],
        entropy_coef: float = 0.01,
        use_baseline: bool = True,
        max_grad_norm: float = 0.5,
        optimizer_type: str = "adam",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.max_grad_norm = max_grad_norm
        
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_layers)
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        elif optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate)
        
        # Running baseline
        self.baseline = 0.0
        self.baseline_alpha = 0.1  # exponential moving average rate
        
        # Logging
        self.training_rewards: List[float] = []
        self.training_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.entropy_values: List[float] = []
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns G_t for each timestep."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """
        Update policy using collected episodes.
        
        Parameters
        ----------
        episodes : list of dicts with keys 'observations', 'actions', 'rewards'
        
        Returns
        -------
        dict with 'policy_loss', 'entropy', 'mean_return'
        """
        total_loss = 0.0
        total_entropy = 0.0
        total_return = 0.0
        total_steps = 0
        
        for episode in episodes:
            obs = torch.FloatTensor(np.array(episode["observations"]))
            acts = torch.LongTensor(episode["actions"])
            returns = self.compute_returns(episode["rewards"])
            
            # Subtract baseline if enabled
            if self.use_baseline:
                ep_return = sum(episode["rewards"])
                advantages = returns - self.baseline
                self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * ep_return
            else:
                advantages = returns
            
            # Compute log probs and entropy
            log_probs, entropy = self.policy.evaluate_action(obs, acts)
            
            # REINFORCE loss: -E[log π(a|s) * advantage]
            policy_loss = -(log_probs * advantages).mean()
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.entropy_coef * entropy_loss
            
            total_loss += policy_loss.item()
            total_entropy += entropy.mean().item()
            total_return += sum(episode["rewards"])
            total_steps += len(episode["rewards"])
        
        # Average over episodes
        n_eps = len(episodes)
        avg_loss = total_loss / n_eps
        
        # Backprop
        self.optimizer.zero_grad()
        
        # Recompute for gradient (using last batch)
        all_obs = torch.FloatTensor(np.concatenate([np.array(e["observations"]) for e in episodes]))
        all_acts = torch.LongTensor(np.concatenate([e["actions"] for e in episodes]))
        all_returns = torch.cat([self.compute_returns(e["rewards"]) for e in episodes])
        
        if self.use_baseline:
            all_advantages = all_returns - self.baseline
        else:
            all_advantages = all_returns
        
        log_probs, entropy = self.policy.evaluate_action(all_obs, all_acts)
        loss = -(log_probs * all_advantages).mean() + self.entropy_coef * (-entropy.mean())
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Log
        self.policy_losses.append(avg_loss)
        self.entropy_values.append(total_entropy / n_eps)
        
        return {
            "policy_loss": avg_loss,
            "entropy": total_entropy / n_eps,
            "mean_return": total_return / n_eps,
            "total_steps": total_steps,
        }
    
    def collect_episode(self, env) -> Tuple[Dict, float, int]:
        """Collect one full episode using current policy."""
        obs, info = env.reset()
        
        observations = []
        actions = []
        rewards = []
        
        done = False
        while not done:
            action, log_prob = self.policy.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            obs = next_obs
            done = terminated or truncated
        
        episode = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
        }
        
        total_reward = sum(rewards)
        length = len(rewards)
        
        self.training_rewards.append(total_reward)
        self.training_lengths.append(length)
        
        return episode, total_reward, length
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.baseline = checkpoint.get("baseline", 0.0)


# ─────────────────────────────────────────────────────────────
# 10 HYPERPARAMETER CONFIGURATIONS
# ─────────────────────────────────────────────────────────────

REINFORCE_EXPERIMENTS = [
    {
        "run_id": 1,
        "description": "Baseline — standard REINFORCE with baseline",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 2,
        "description": "No baseline — pure REINFORCE (high variance)",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.01,
            "use_baseline": False, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 3,
        "description": "Low LR (1e-4) — slower but stabler",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 4,
        "description": "High LR (5e-3) — fast learning test",
        "hyperparams": {
            "learning_rate": 5e-3, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 5,
        "description": "Low gamma (0.95) — short-term focus",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.95,
            "hidden_layers": [256, 256], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 6,
        "description": "High entropy (0.05) — encourage exploration",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.05,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 7,
        "description": "Deep network (3×256) — more capacity",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99,
            "hidden_layers": [256, 256, 256], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 8,
        "description": "Small network (128,64) + more episodes per update (10)",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99,
            "hidden_layers": [128, 64], "entropy_coef": 0.01,
            "use_baseline": True, "episodes_per_update": 10,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
    {
        "run_id": 9,
        "description": "RMSprop optimizer + low entropy",
        "hyperparams": {
            "learning_rate": 5e-4, "gamma": 0.99,
            "hidden_layers": [256, 256], "entropy_coef": 0.001,
            "use_baseline": True, "episodes_per_update": 5,
            "max_grad_norm": 1.0, "optimizer": "rmsprop",
        },
    },
    {
        "run_id": 10,
        "description": "Large batch (15 episodes) + high gamma (0.999)",
        "hyperparams": {
            "learning_rate": 5e-4, "gamma": 0.999,
            "hidden_layers": [256, 128], "entropy_coef": 0.02,
            "use_baseline": True, "episodes_per_update": 15,
            "max_grad_norm": 0.5, "optimizer": "adam",
        },
    },
]


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train_reinforce_experiment(experiment: dict, total_timesteps: int = TOTAL_TIMESTEPS):
    run_id = experiment["run_id"]
    hp = experiment["hyperparams"]
    
    print(f"\n{'='*70}")
    print(f"  REINFORCE RUN {run_id}/10: {experiment['description']}")
    print(f"  {total_timesteps:,} timesteps | hidden={hp['hidden_layers']} | baseline={hp['use_baseline']}")
    print(f"{'='*70}")
    
    env = make_env(seed=run_id * 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = REINFORCE(
        obs_dim=obs_dim, act_dim=act_dim,
        learning_rate=hp["learning_rate"], gamma=hp["gamma"],
        hidden_layers=hp["hidden_layers"], entropy_coef=hp["entropy_coef"],
        use_baseline=hp["use_baseline"], max_grad_norm=hp["max_grad_norm"],
        optimizer_type=hp["optimizer"],
    )
    
    episodes_per_update = hp["episodes_per_update"]
    total_steps = 0
    update_count = 0
    eval_results = []
    
    start = time.time()
    
    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_timesteps, desc=f"  REINFORCE Run {run_id}", unit=" steps")
    except ImportError:
        pbar = None
    
    while total_steps < total_timesteps:
        # Collect batch of episodes
        batch = []
        batch_steps = 0
        for _ in range(episodes_per_update):
            episode, ep_reward, ep_len = agent.collect_episode(env)
            batch.append(episode)
            batch_steps += ep_len
            total_steps += ep_len
            
            if pbar:
                pbar.update(ep_len)
            
            if total_steps >= total_timesteps:
                break
        
        # Update policy
        update_info = agent.update(batch)
        update_count += 1
        
        # Progress logging
        if update_count % 10 == 0:
            recent_rewards = agent.training_rewards[-episodes_per_update*10:]
            mean_r = np.mean(recent_rewards) if recent_rewards else 0
            print(f"  [{total_steps:>7,d}] updates={update_count} "
                  f"mean_reward={mean_r:+.2f} "
                  f"entropy={update_info['entropy']:.3f} "
                  f"loss={update_info['policy_loss']:.4f}")
        
        # Periodic evaluation
        if update_count % 20 == 0:
            metrics = evaluate_reinforce(agent.policy, n_episodes=5, seed=9999)
            metrics["timestep"] = total_steps
            eval_results.append(metrics)
            print(f"  [EVAL] reward={metrics['mean_reward']:+.2f} "
                  f"survival={metrics['survival_rate']:.0%}")
    
    train_time = time.time() - start
    
    if pbar:
        pbar.close()
    
    # Final evaluation
    print(f"\n  Final evaluation run {run_id}...")
    metrics = evaluate_reinforce(agent.policy, n_episodes=20, seed=5000)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "pg", f"reinforce_run_{run_id}.pt")
    agent.save(model_path)
    
    # Save curves
    curves = {
        "episode_rewards": agent.training_rewards,
        "episode_lengths": agent.training_lengths,
        "policy_losses": agent.policy_losses,
        "entropy_values": agent.entropy_values,
        "eval_results": eval_results,
    }
    with open(os.path.join(RESULTS_DIR, "reinforce", f"reinforce_run_{run_id}_curves.json"), "w") as f:
        json.dump(curves, f, indent=2, default=str)
    
    env.close()
    return metrics, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    
    logger = ExperimentLogger("reinforce")
    exps = REINFORCE_EXPERIMENTS if args.run is None else [e for e in REINFORCE_EXPERIMENTS if e["run_id"] == args.run]
    
    print(f"\n  REINFORCE TRAINING — {len(exps)} experiments × {args.timesteps:,} timesteps\n")
    
    best_reward, best_run = -float("inf"), -1
    
    for exp in exps:
        metrics, train_time = train_reinforce_experiment(exp, args.timesteps)
        logger.log_experiment(exp["run_id"], exp["hyperparams"], metrics, train_time, args.timesteps, exp["description"])
        if metrics["mean_reward"] > best_reward:
            best_reward, best_run = metrics["mean_reward"], exp["run_id"]
    
    if best_run > 0:
        import shutil
        src = os.path.join(MODELS_DIR, "pg", f"reinforce_run_{best_run}.pt")
        dst = os.path.join(MODELS_DIR, "pg", "best_reinforce.pt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"\nBest REINFORCE: Run {best_run} ({best_reward:+.2f}) → {dst}")
    
    logger.print_summary_table()


if __name__ == "__main__":
    main()