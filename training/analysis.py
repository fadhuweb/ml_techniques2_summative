"""
Analysis & Plotting Script
============================

Generates all required visualizations for the report:
1. Cumulative reward curves for all methods (subplots)
2. DQN objective/loss curves (TD loss over time)
3. Policy gradient entropy curves (exploration decay)
4. Convergence plots (when each method stabilizes)
5. Generalization test (different climate seeds)
6. Algorithm comparison bar charts
7. Per-zone population trajectories for best models

Usage:
    python -m training.analysis                    # Generate all plots
    python -m training.analysis --plot rewards      # Generate specific plot

Outputs saved to: results/plots/
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent styling
COLORS = {
    "dqn": "#2196F3",
    "reinforce": "#FF9800",
    "ppo": "#4CAF50",
}
ALGO_LABELS = {
    "dqn": "DQN (Value-Based)",
    "reinforce": "REINFORCE (Policy Gradient)",
    "ppo": "PPO (Policy Gradient)",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_experiment_results(algorithm: str):
    """Load the experiment summary JSON for an algorithm. Forces numeric conversion."""
    path = os.path.join(RESULTS_DIR, algorithm, f"{algorithm}_experiments.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        
        # Force numeric fields to float
        for entry in data:
            for key, val in entry.items():
                if key.startswith(("eval_", "hp_", "training_time")):
                    try:
                        entry[key] = float(val)
                    except (ValueError, TypeError):
                        pass  # keep as-is for strings like policy_kwargs
        return data
    print(f"  Warning: {path} not found")
    return []


def load_training_curves(algorithm: str, run_id: int):
    """Load training curves JSON for a specific run. Forces numeric conversion."""
    path = os.path.join(RESULTS_DIR, algorithm, f"{algorithm}_run_{run_id}_curves.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        
        # Force all list values to float (fixes string contamination from json default=str)
        for key in data:
            if isinstance(data[key], list) and data[key]:
                if isinstance(data[key][0], (int, float, str)):
                    try:
                        data[key] = [float(x) for x in data[key]]
                    except (ValueError, TypeError):
                        pass  # skip non-numeric lists like eval_results
        return data
    return None


def get_best_run_id(algorithm: str):
    """Find the best run ID from experiment results."""
    results = load_experiment_results(algorithm)
    if not results:
        return 1
    best = max(results, key=lambda r: r.get("eval_mean_reward", -9999))
    return best.get("run_id", 1)


def smooth(values, window=10):
    """Apply moving average smoothing."""
    if not values or len(values) < window:
        return [float(v) for v in values] if values else []
    arr = np.array([float(v) for v in values], dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid").tolist()


# ─────────────────────────────────────────────────────────────
# PLOT 1: CUMULATIVE REWARD CURVES (all methods, subplots)
# ─────────────────────────────────────────────────────────────

def plot_cumulative_rewards():
    """
    Generate cumulative reward curves for all 3 methods as subplots.
    Shows the best run for each algorithm + shaded region for all runs.
    """
    print("  Generating cumulative reward curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Cumulative Reward Curves — All Methods", fontsize=15, fontweight="bold", y=1.02)
    
    for idx, algo in enumerate(["dqn", "reinforce", "ppo"]):
        ax = axes[idx]
        results = load_experiment_results(algo)
        best_id = get_best_run_id(algo)
        
        all_rewards = []
        
        # Plot all runs as faint lines
        for run in range(1, 11):
            curves = load_training_curves(algo, run)
            if curves and "episode_rewards" in curves:
                rewards = curves["episode_rewards"]
                if len(rewards) > 0:
                    cumulative = np.cumsum(rewards)
                    episodes = np.arange(1, len(cumulative) + 1)
                    
                    # Smooth for readability
                    smoothed = smooth(rewards, window=20)
                    cum_smoothed = np.cumsum(smoothed)
                    
                    alpha = 0.15 if run != best_id else 0.0
                    if alpha > 0:
                        ax.plot(range(len(smoothed)), smoothed, 
                                color=COLORS[algo], alpha=alpha, linewidth=0.8)
                    all_rewards.append(rewards)
        
        # Plot best run prominently
        best_curves = load_training_curves(algo, best_id)
        if best_curves and "episode_rewards" in best_curves:
            rewards = best_curves["episode_rewards"]
            smoothed = smooth(rewards, window=20)
            ax.plot(range(len(smoothed)), smoothed,
                    color=COLORS[algo], linewidth=2.0, label=f"Best (Run {best_id})")
        
        # Formatting
        ax.set_title(ALGO_LABELS[algo], color=COLORS[algo], fontweight="bold")
        ax.set_xlabel("Episode")
        if idx == 0:
            ax.set_ylabel("Episode Reward (smoothed)")
        ax.legend(loc="lower right")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_cumulative_reward_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 2: DQN OBJECTIVE/LOSS CURVES
# ─────────────────────────────────────────────────────────────

def plot_dqn_loss_curves():
    """
    Plot DQN-specific metrics: TD loss and Q-value estimates over training.
    Uses evaluation reward as proxy for Q-value quality.
    """
    print("  Generating DQN objective curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN Objective Curves", fontsize=15, fontweight="bold", y=1.02)
    
    best_id = get_best_run_id("dqn")
    
    # Left: Episode rewards over training (proxy for Q-value convergence)
    ax1 = axes[0]
    for run in range(1, 11):
        curves = load_training_curves("dqn", run)
        if curves and "episode_rewards" in curves:
            rewards = curves["episode_rewards"]
            smoothed = smooth(rewards, window=30)
            alpha = 0.3 if run != best_id else 1.0
            lw = 0.8 if run != best_id else 2.0
            label = f"Run {run}" if run == best_id else None
            ax1.plot(range(len(smoothed)), smoothed,
                     color=COLORS["dqn"], alpha=alpha, linewidth=lw, label=label)
    
    ax1.set_title("Training Reward (Q-value convergence proxy)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward (smoothed)")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Right: Evaluation metrics over timesteps
    ax2 = axes[1]
    for run in range(1, 11):
        curves = load_training_curves("dqn", run)
        if curves and "eval_results" in curves:
            evals = curves["eval_results"]
            if evals:
                timesteps = [e.get("timestep", i*50000) for i, e in enumerate(evals)]
                eval_rewards = [e["mean_reward"] for e in evals]
                alpha = 0.3 if run != best_id else 1.0
                lw = 0.8 if run != best_id else 2.0
                label = f"Run {run} (best)" if run == best_id else None
                ax2.plot(timesteps, eval_rewards,
                         color=COLORS["dqn"], alpha=alpha, linewidth=lw,
                         marker="o" if run == best_id else None, markersize=4, label=label)
    
    ax2.set_title("Evaluation Reward vs Training Timesteps")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Eval Reward (20 episodes)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_dqn_objective_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 3: POLICY GRADIENT ENTROPY CURVES
# ─────────────────────────────────────────────────────────────

def plot_entropy_curves():
    """
    Plot entropy curves for REINFORCE and PPO showing exploration decay.
    """
    print("  Generating entropy curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Policy Gradient Entropy Curves — Exploration Decay", fontsize=15, fontweight="bold", y=1.02)
    
    # REINFORCE entropy
    ax1 = axes[0]
    best_reinforce = get_best_run_id("reinforce")
    for run in range(1, 11):
        curves = load_training_curves("reinforce", run)
        if curves and "entropy_values" in curves:
            entropy = curves["entropy_values"]
            smoothed = smooth(entropy, window=10)
            alpha = 0.25 if run != best_reinforce else 1.0
            lw = 0.8 if run != best_reinforce else 2.0
            label = f"Run {run} (best)" if run == best_reinforce else None
            ax1.plot(range(len(smoothed)), smoothed,
                     color=COLORS["reinforce"], alpha=alpha, linewidth=lw, label=label)
    
    ax1.set_title("REINFORCE Entropy", color=COLORS["reinforce"], fontweight="bold")
    ax1.set_xlabel("Policy Update")
    ax1.set_ylabel("Entropy")
    ax1.legend()
    
    # PPO entropy (from eval results)
    ax2 = axes[1]
    best_ppo = get_best_run_id("ppo")
    for run in range(1, 11):
        curves = load_training_curves("ppo", run)
        if curves and "episode_rewards" in curves:
            # PPO doesn't directly log entropy in our callback,
            # so we use reward variance as an exploration proxy
            rewards = curves["episode_rewards"]
            if len(rewards) > 20:
                # Rolling standard deviation as exploration proxy
                window = 20
                rolling_std = []
                for i in range(window, len(rewards)):
                    rolling_std.append(np.std(rewards[i-window:i]))
                
                smoothed = smooth(rolling_std, window=10)
                alpha = 0.25 if run != best_ppo else 1.0
                lw = 0.8 if run != best_ppo else 2.0
                label = f"Run {run} (best)" if run == best_ppo else None
                ax2.plot(range(len(smoothed)), smoothed,
                         color=COLORS["ppo"], alpha=alpha, linewidth=lw, label=label)
    
    ax2.set_title("PPO Reward Variance (exploration proxy)", color=COLORS["ppo"], fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rolling Std Dev of Reward")
    ax2.legend()
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_entropy_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 4: CONVERGENCE PLOTS
# ─────────────────────────────────────────────────────────────

def plot_convergence():
    """
    Plot convergence comparison — all 3 methods on one chart showing
    when each method stabilizes (evaluation reward over timesteps).
    """
    print("  Generating convergence plots...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Convergence Comparison — All Methods", fontsize=15, fontweight="bold", y=1.02)
    
    for algo in ["dqn", "reinforce", "ppo"]:
        best_id = get_best_run_id(algo)
        curves = load_training_curves(algo, best_id)
        
        if curves and "eval_results" in curves:
            evals = curves["eval_results"]
            if evals:
                timesteps = [e.get("timestep", i * 50000) for i, e in enumerate(evals)]
                eval_rewards = [e["mean_reward"] for e in evals]
                ax.plot(timesteps, eval_rewards,
                        color=COLORS[algo], linewidth=2.5,
                        marker="o", markersize=5,
                        label=f"{ALGO_LABELS[algo]} (Run {best_id})")
        
        # For REINFORCE, use training rewards mapped to timesteps
        elif curves and "episode_rewards" in curves:
            rewards = curves["episode_rewards"]
            # Estimate timesteps (each episode ~120 steps in our env)
            avg_ep_len = 80  # approximate
            chunk_size = max(1, 50000 // avg_ep_len)
            
            eval_points_ts = []
            eval_points_r = []
            for i in range(0, len(rewards), chunk_size):
                chunk = rewards[i:i+chunk_size]
                if chunk:
                    eval_points_ts.append(i * avg_ep_len)
                    eval_points_r.append(np.mean(chunk))
            
            if eval_points_ts:
                ax.plot(eval_points_ts, eval_points_r,
                        color=COLORS[algo], linewidth=2.5,
                        marker="s", markersize=5,
                        label=f"{ALGO_LABELS[algo]} (Run {best_id})")
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Zero reward baseline")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_convergence_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 5: GENERALIZATION TEST
# ─────────────────────────────────────────────────────────────

def plot_generalization_test():
    """
    Test best models on modified environments (different seeds, difficulty)
    to assess generalization.
    """
    print("  Running generalization tests...")
    
    from training.utils import make_env, evaluate_model, evaluate_reinforce
    
    # Test conditions
    test_configs = [
        {"label": "Training seed", "seed": 1000, "difficulty": "normal"},
        {"label": "New seed A", "seed": 7777, "difficulty": "normal"},
        {"label": "New seed B", "seed": 3333, "difficulty": "normal"},
        {"label": "Easy climate", "seed": 1000, "difficulty": "easy"},
        {"label": "Hard climate", "seed": 1000, "difficulty": "hard"},
    ]
    
    results = {algo: [] for algo in ["dqn", "ppo", "reinforce"]}
    
    for algo in ["dqn", "ppo"]:
        # Load SB3 model
        model_paths = {
            "dqn": "models/dqn/best_dqn.zip",
            "ppo": "models/pg/best_ppo.zip",
        }
        path = model_paths[algo]
        if not os.path.exists(path):
            print(f"    Skipping {algo} — {path} not found")
            continue
        
        if algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(path)
        
        for config in test_configs:
            env = make_env(seed=config["seed"], difficulty=config["difficulty"])
            metrics = evaluate_model(model, n_episodes=10, seed=config["seed"])
            results[algo].append({
                "label": config["label"],
                "mean_reward": metrics["mean_reward"],
                "std_reward": metrics["std_reward"],
                "survival_rate": metrics["survival_rate"],
            })
            env.close()
            print(f"    {algo} on {config['label']}: reward={metrics['mean_reward']:+.1f} survival={metrics['survival_rate']:.0%}")
    
    # REINFORCE
    reinforce_path = "models/pg/best_reinforce.pt"
    if os.path.exists(reinforce_path):
        import torch
        from training.reinforce_training import REINFORCE
        
        env_temp = make_env()
        agent = REINFORCE(
            obs_dim=env_temp.observation_space.shape[0],
            act_dim=env_temp.action_space.n,
        )
        agent.load(reinforce_path)
        env_temp.close()
        
        for config in test_configs:
            metrics = evaluate_reinforce(agent.policy, n_episodes=10, seed=config["seed"])
            results["reinforce"].append({
                "label": config["label"],
                "mean_reward": metrics["mean_reward"],
                "std_reward": metrics["std_reward"],
                "survival_rate": metrics["survival_rate"],
            })
            print(f"    reinforce on {config['label']}: reward={metrics['mean_reward']:+.1f} survival={metrics['survival_rate']:.0%}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Generalization Test — Performance on Unseen Conditions", fontsize=15, fontweight="bold", y=1.02)
    
    labels = [c["label"] for c in test_configs]
    x = np.arange(len(labels))
    bar_width = 0.25
    
    # Reward bars
    ax1 = axes[0]
    for i, algo in enumerate(["dqn", "reinforce", "ppo"]):
        if results[algo]:
            means = [r["mean_reward"] for r in results[algo]]
            stds = [r["std_reward"] for r in results[algo]]
            ax1.bar(x + i * bar_width, means, bar_width, yerr=stds,
                    label=ALGO_LABELS[algo], color=COLORS[algo], alpha=0.85,
                    capsize=3)
    
    ax1.set_title("Mean Reward Across Test Conditions")
    ax1.set_xlabel("Test Condition")
    ax1.set_ylabel("Mean Reward")
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Survival rate bars
    ax2 = axes[1]
    for i, algo in enumerate(["dqn", "reinforce", "ppo"]):
        if results[algo]:
            survivals = [r["survival_rate"] * 100 for r in results[algo]]
            ax2.bar(x + i * bar_width, survivals, bar_width,
                    label=ALGO_LABELS[algo], color=COLORS[algo], alpha=0.85)
    
    ax2.set_title("Survival Rate Across Test Conditions")
    ax2.set_xlabel("Test Condition")
    ax2.set_ylabel("Survival Rate (%)")
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0, 105)
    ax2.legend()
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_generalization_test.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")
    
    # Save raw results
    with open(os.path.join(PLOTS_DIR, "generalization_results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ─────────────────────────────────────────────────────────────
# PLOT 6: ALGORITHM COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_algorithm_comparison():
    """
    Bar chart comparing best run from each algorithm across key metrics.
    """
    print("  Generating algorithm comparison...")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Algorithm Comparison — Best Run from Each Method", fontsize=15, fontweight="bold", y=1.02)
    
    metrics_to_plot = [
        ("eval_mean_reward", "Mean Reward", axes[0]),
        ("eval_survival_rate", "Survival Rate", axes[1]),
        ("eval_mean_final_pop", "Mean Final Population", axes[2]),
        ("training_time_seconds", "Training Time (s)", axes[3]),
    ]
    
    algos = ["dqn", "reinforce", "ppo"]
    best_runs = {}
    
    for algo in algos:
        results = load_experiment_results(algo)
        if results:
            best = max(results, key=lambda r: r.get("eval_mean_reward", -9999))
            best_runs[algo] = best
    
    for metric_key, metric_label, ax in metrics_to_plot:
        values = []
        colors = []
        labels = []
        
        for algo in algos:
            if algo in best_runs:
                val = best_runs[algo].get(metric_key, 0)
                values.append(val)
                colors.append(COLORS[algo])
                labels.append(algo.upper())
        
        if values:
            bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
            ax.set_title(metric_label)
            ax.set_ylabel(metric_label)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                fmt = f"{val:.0%}" if "rate" in metric_key else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(abs(v) for v in values),
                        fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_algorithm_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 7: HYPERPARAMETER SENSITIVITY HEATMAPS
# ─────────────────────────────────────────────────────────────

def plot_hyperparameter_sensitivity():
    """
    Show how key hyperparameters affected performance for each algorithm.
    """
    print("  Generating hyperparameter sensitivity plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Hyperparameter Sensitivity — Reward vs Key Hyperparameters", fontsize=15, fontweight="bold", y=1.02)
    
    # DQN: learning rate vs reward
    ax1 = axes[0]
    dqn_results = load_experiment_results("dqn")
    if dqn_results:
        lrs = [r.get("hp_learning_rate", 0) for r in dqn_results]
        rewards = [r.get("eval_mean_reward", 0) for r in dqn_results]
        run_ids = [r.get("run_id", 0) for r in dqn_results]
        
        ax1.scatter(lrs, rewards, c=COLORS["dqn"], s=80, alpha=0.8, edgecolors="white", linewidth=1.5)
        for lr, rew, rid in zip(lrs, rewards, run_ids):
            ax1.annotate(f"R{rid}", (lr, rew), fontsize=7, ha="center", va="bottom")
        
        ax1.set_title("DQN: Learning Rate Impact", color=COLORS["dqn"], fontweight="bold")
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Mean Eval Reward")
        ax1.set_xscale("log")
    
    # REINFORCE: learning rate vs reward
    ax2 = axes[1]
    reinforce_results = load_experiment_results("reinforce")
    if reinforce_results:
        lrs = [r.get("hp_learning_rate", 0) for r in reinforce_results]
        rewards = [r.get("eval_mean_reward", 0) for r in reinforce_results]
        run_ids = [r.get("run_id", 0) for r in reinforce_results]
        
        ax2.scatter(lrs, rewards, c=COLORS["reinforce"], s=80, alpha=0.8, edgecolors="white", linewidth=1.5)
        for lr, rew, rid in zip(lrs, rewards, run_ids):
            ax2.annotate(f"R{rid}", (lr, rew), fontsize=7, ha="center", va="bottom")
        
        ax2.set_title("REINFORCE: Learning Rate Impact", color=COLORS["reinforce"], fontweight="bold")
        ax2.set_xlabel("Learning Rate")
        ax2.set_ylabel("Mean Eval Reward")
        ax2.set_xscale("log")
    
    # PPO: clip range vs reward
    ax3 = axes[2]
    ppo_results = load_experiment_results("ppo")
    if ppo_results:
        clips = [r.get("hp_clip_range", 0.2) for r in ppo_results]
        rewards = [r.get("eval_mean_reward", 0) for r in ppo_results]
        run_ids = [r.get("run_id", 0) for r in ppo_results]
        
        ax3.scatter(clips, rewards, c=COLORS["ppo"], s=80, alpha=0.8, edgecolors="white", linewidth=1.5)
        for cl, rew, rid in zip(clips, rewards, run_ids):
            ax3.annotate(f"R{rid}", (cl, rew), fontsize=7, ha="center", va="bottom")
        
        ax3.set_title("PPO: Clip Range Impact", color=COLORS["ppo"], fontweight="bold")
        ax3.set_xlabel("Clip Range")
        ax3.set_ylabel("Mean Eval Reward")
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "07_hyperparameter_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 8: BEST MODEL — ZONE POPULATION TRAJECTORIES
# ─────────────────────────────────────────────────────────────

def plot_best_model_trajectory():
    """
    Run the best overall model for one full episode and plot
    per-zone population and habitat trajectories.
    """
    print("  Generating best model trajectory...")
    
    from training.utils import make_env
    
    # Find best overall model
    best_algo = None
    best_reward = -float("inf")
    
    for algo in ["dqn", "ppo"]:
        results = load_experiment_results(algo)
        if results:
            best = max(results, key=lambda r: r.get("eval_mean_reward", -9999))
            if best.get("eval_mean_reward", -9999) > best_reward:
                best_reward = best["eval_mean_reward"]
                best_algo = algo
    
    if best_algo is None:
        print("    No trained models found. Skipping.")
        return
    
    # Load model
    if best_algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load("models/dqn/best_dqn.zip")
    else:
        from stable_baselines3 import PPO
        model = PPO.load("models/pg/best_ppo.zip")
    
    # Run one episode
    env = make_env(seed=42)
    obs, info = env.reset()
    
    from environment.world_model import ZONES
    zone_pops = {z.name: [] for z in ZONES}
    zone_habs = {z.name: [] for z in ZONES}
    
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        
        for i, z in enumerate(ZONES):
            zone_pops[z.name].append(env.zone_states[i]["wildlife_pop"])
            zone_habs[z.name].append(env.zone_states[i]["habitat_integrity"])
        
        done = terminated or truncated
    
    env.close()
    
    # Plot
    zone_colors = ["#2196F3", "#E91E63", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Best Model ({best_algo.upper()}) — Full Episode Trajectory", fontsize=15, fontweight="bold", y=1.01)
    
    # Population trajectories
    ax1 = axes[0]
    for i, (z_name, pops) in enumerate(zone_pops.items()):
        ax1.plot(pops, color=zone_colors[i % len(zone_colors)], linewidth=1.8, label=z_name)
    
    ax1.set_title("Wildlife Population per Zone")
    ax1.set_ylabel("Population Index")
    ax1.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Extinction threshold")
    ax1.legend(loc="upper right", ncol=3, fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    
    # Habitat trajectories
    ax2 = axes[1]
    for i, (z_name, habs) in enumerate(zone_habs.items()):
        ax2.plot(habs, color=zone_colors[i % len(zone_colors)], linewidth=1.8, label=z_name)
    
    ax2.set_title("Habitat Integrity per Zone")
    ax2.set_xlabel("Month (timestep)")
    ax2.set_ylabel("Habitat Integrity")
    ax2.legend(loc="upper right", ncol=3, fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_best_model_trajectory.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

PLOT_FUNCTIONS = {
    "rewards": plot_cumulative_rewards,
    "dqn_loss": plot_dqn_loss_curves,
    "entropy": plot_entropy_curves,
    "convergence": plot_convergence,
    "generalization": plot_generalization_test,
    "comparison": plot_algorithm_comparison,
    "sensitivity": plot_hyperparameter_sensitivity,
    "trajectory": plot_best_model_trajectory,
}


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument("--plot", type=str, default=None,
                        choices=list(PLOT_FUNCTIONS.keys()),
                        help="Generate a specific plot only")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ANALYSIS — Generating Report Plots")
    print(f"  Output: {PLOTS_DIR}/")
    print("=" * 60)
    
    if args.plot:
        PLOT_FUNCTIONS[args.plot]()
    else:
        for name, func in PLOT_FUNCTIONS.items():
            try:
                func()
            except Exception as e:
                print(f"    ERROR in {name}: {e}")
    
    print(f"\n  All plots saved to {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()