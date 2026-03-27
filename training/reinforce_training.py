import os, sys, time, json, argparse, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from training.utils import (
    make_env, evaluate_model, ExperimentLogger,
    TrainingMetricsCallback, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
)

TOTAL_TIMESTEPS = 500_000

PPO_EXPERIMENTS = [
    {
        "run_id": 1,
        "description": "Baseline — default PPO",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 2,
        "description": "High LR (1e-3) — fast learning",
        "hyperparams": {
            "learning_rate": 1e-3, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 3,
        "description": "Low LR (5e-5) — slow stable training",
        "hyperparams": {
            "learning_rate": 5e-5, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 4,
        "description": "Tight clip (0.1) — conservative policy updates",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.1,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 5,
        "description": "Wide clip (0.3) + more epochs (20) — aggressive updates",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.3,
            "n_epochs": 20, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 6,
        "description": "High entropy (0.05) — encourage exploration",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.05, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 7,
        "description": "Low gamma (0.95) + low GAE lambda (0.9) — short horizon",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.95, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.90, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 8,
        "description": "Large rollout (4096 steps) + large batch (128)",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 128, "n_steps": 4096,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
    {
        "run_id": 9,
        "description": "Deep network (3 layers) + high vf_coef (1.0)",
        "hyperparams": {
            "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2,
            "n_epochs": 10, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}},
        },
    },
    {
        "run_id": 10,
        "description": "Tuned — mid LR, high gamma, moderate clip, balanced",
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.995, "clip_range": 0.15,
            "n_epochs": 15, "batch_size": 64, "n_steps": 2048,
            "gae_lambda": 0.98, "ent_coef": 0.02, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        },
    },
]


def train_ppo_experiment(experiment: dict, total_timesteps: int = TOTAL_TIMESTEPS):
    run_id = experiment["run_id"]
    hp = experiment["hyperparams"]
    
    print(f"\n{'='*70}")
    print(f"  PPO RUN {run_id}/10: {experiment['description']}")
    print(f"  {total_timesteps:,} timesteps | clip={hp['clip_range']} | epochs={hp['n_epochs']}")
    print(f"{'='*70}")
    
    env = Monitor(make_env(seed=run_id * 100))
    
    model = PPO(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], gamma=hp["gamma"],
        clip_range=hp["clip_range"], n_epochs=hp["n_epochs"],
        batch_size=hp["batch_size"], n_steps=hp["n_steps"],
        gae_lambda=hp["gae_lambda"], ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
        policy_kwargs=hp.get("policy_kwargs", {}),
        tensorboard_log=os.path.join(LOGS_DIR, "ppo"),
        verbose=0, seed=run_id,
    )
    
    callback = TrainingMetricsCallback(eval_freq=50_000, n_eval_episodes=5, verbose=1)
    
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback,
                tb_log_name=f"ppo_run_{run_id}", progress_bar=True)
    train_time = time.time() - start
    
    print(f"\n  Evaluating run {run_id}...")
    metrics = evaluate_model(model, n_episodes=20, seed=5000)
    
    model.save(os.path.join(MODELS_DIR, "pg", f"ppo_run_{run_id}"))
    
    curves = callback.get_training_curves()
    with open(os.path.join(RESULTS_DIR, "ppo", f"ppo_run_{run_id}_curves.json"), "w") as f:
        json.dump(curves, f, indent=2, default=str)
    
    env.close()
    return metrics, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    
    logger = ExperimentLogger("ppo")
    exps = PPO_EXPERIMENTS if args.run is None else [e for e in PPO_EXPERIMENTS if e["run_id"] == args.run]
    
    print(f"\n  PPO TRAINING — {len(exps)} experiments × {args.timesteps:,} timesteps\n")
    
    best_reward, best_run = -float("inf"), -1
    
    for exp in exps:
        metrics, train_time = train_ppo_experiment(exp, args.timesteps)
        hp_flat = {k: str(v) if k == "policy_kwargs" else v for k, v in exp["hyperparams"].items()}
        logger.log_experiment(exp["run_id"], hp_flat, metrics, train_time, args.timesteps, exp["description"])
        if metrics["mean_reward"] > best_reward:
            best_reward, best_run = metrics["mean_reward"], exp["run_id"]
    
    if best_run > 0:
        src = os.path.join(MODELS_DIR, "pg", f"ppo_run_{best_run}.zip")
        dst = os.path.join(MODELS_DIR, "pg", "best_ppo.zip")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"\nBest PPO: Run {best_run} ({best_reward:+.2f}) → {dst}")
    
    logger.print_summary_table()


if __name__ == "__main__":
    main()