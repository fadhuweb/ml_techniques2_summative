import os, sys, time, json, argparse, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from training.utils import (
    make_env, evaluate_model, ExperimentLogger,
    TrainingMetricsCallback, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
)

TOTAL_TIMESTEPS = 500_000

DQN_EXPERIMENTS = [
    {
        "run_id": 1,
        "description": "Baseline — default DQN",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 2,
        "description": "High LR (1e-3) — fast convergence test",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 3,
        "description": "Low LR (5e-5) — stability test",
        "hyperparams": {
            "learning_rate": 5e-5, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 4,
        "description": "Low gamma (0.95) — short planning horizon",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.95, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 5,
        "description": "Long exploration (0.5) — extended explore phase",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 6,
        "description": "Large buffer (200K) + large batch (128)",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 200_000,
            "batch_size": 128, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 10000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 7,
        "description": "Deep network (3×256) — more capacity",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 64, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 1000, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256, 256]},
        },
    },
    {
        "run_id": 8,
        "description": "Small net (128,128) + frequent target update (500)",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000,
            "batch_size": 32, "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "target_update_interval": 500, "train_freq": 4,
            "gradient_steps": 1, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [128, 128]},
        },
    },
    {
        "run_id": 9,
        "description": "Aggressive — high LR, low gamma, fast explore",
        "hyperparams": {
            "learning_rate": 5e-4, "gamma": 0.97, "buffer_size": 100_000,
            "batch_size": 64, "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.1,
            "target_update_interval": 750, "train_freq": 4,
            "gradient_steps": 2, "learning_starts": 5000,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
    {
        "run_id": 10,
        "description": "Conservative — low LR, high gamma, long explore, big buffer",
        "hyperparams": {
            "learning_rate": 3e-5, "gamma": 0.995, "buffer_size": 200_000,
            "batch_size": 128, "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01,
            "target_update_interval": 2000, "train_freq": 8,
            "gradient_steps": 1, "learning_starts": 10000,
            "policy_kwargs": {"net_arch": [256, 128]},
        },
    },
]


def train_dqn_experiment(experiment: dict, total_timesteps: int = TOTAL_TIMESTEPS):
    run_id = experiment["run_id"]
    hp = experiment["hyperparams"]
    
    print(f"\n{'='*70}")
    print(f"  DQN RUN {run_id}/10: {experiment['description']}")
    print(f"  {total_timesteps:,} timesteps | net_arch={hp['policy_kwargs']['net_arch']}")
    print(f"{'='*70}")
    
    env = Monitor(make_env(seed=run_id * 100))
    
    model = DQN(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], gamma=hp["gamma"],
        buffer_size=hp["buffer_size"], batch_size=hp["batch_size"],
        exploration_fraction=hp["exploration_fraction"],
        exploration_initial_eps=hp["exploration_initial_eps"],
        exploration_final_eps=hp["exploration_final_eps"],
        target_update_interval=hp["target_update_interval"],
        train_freq=hp["train_freq"], gradient_steps=hp["gradient_steps"],
        learning_starts=hp["learning_starts"],
        policy_kwargs=hp.get("policy_kwargs", {}),
        tensorboard_log=os.path.join(LOGS_DIR, "dqn"),
        verbose=0, seed=run_id,
    )
    
    callback = TrainingMetricsCallback(eval_freq=50_000, n_eval_episodes=5, verbose=1)
    
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback,
                tb_log_name=f"dqn_run_{run_id}", progress_bar=True)
    train_time = time.time() - start
    
    print(f"\n  Evaluating run {run_id}...")
    metrics = evaluate_model(model, n_episodes=20, seed=5000)
    
    model.save(os.path.join(MODELS_DIR, "dqn", f"dqn_run_{run_id}"))
    
    curves = callback.get_training_curves()
    with open(os.path.join(RESULTS_DIR, "dqn", f"dqn_run_{run_id}_curves.json"), "w") as f:
        json.dump(curves, f, indent=2, default=str)
    
    env.close()
    return metrics, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    
    logger = ExperimentLogger("dqn")
    exps = DQN_EXPERIMENTS if args.run is None else [e for e in DQN_EXPERIMENTS if e["run_id"] == args.run]
    
    print(f"\n  DQN TRAINING — {len(exps)} experiments × {args.timesteps:,} timesteps\n")
    
    best_reward, best_run = -float("inf"), -1
    
    for exp in exps:
        metrics, train_time = train_dqn_experiment(exp, args.timesteps)
        hp_flat = {k: str(v) if k == "policy_kwargs" else v for k, v in exp["hyperparams"].items()}
        logger.log_experiment(exp["run_id"], hp_flat, metrics, train_time, args.timesteps, exp["description"])
        if metrics["mean_reward"] > best_reward:
            best_reward, best_run = metrics["mean_reward"], exp["run_id"]
    
    if best_run > 0:
        src = os.path.join(MODELS_DIR, "dqn", f"dqn_run_{best_run}.zip")
        dst = os.path.join(MODELS_DIR, "dqn", "best_dqn.zip")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"\nBest DQN: Run {best_run} ({best_reward:+.2f}) → {dst}")
    
    logger.print_summary_table()


if __name__ == "__main__":
    main()