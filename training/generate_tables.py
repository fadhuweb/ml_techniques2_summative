"""
Hyperparameter Tables Generator
=================================

Reads the experiment CSV files and generates formatted tables
with analysis paragraphs for the report.

Outputs:
- results/tables/hyperparameter_tables.md (for report)
- results/tables/hyperparameter_tables.html (visual preview)

Usage:
    python -m training.generate_tables
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)


def read_csv(path):
    """Read CSV and return list of dicts."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return []
    # Try utf-8 first, fall back to cp1252 for Windows-generated CSVs
    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            with open(path, newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    print(f"  Error: Could not decode {path}")
    return []


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fmt(val, decimals=2):
    """Format a number nicely."""
    v = safe_float(val)
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.{decimals}f}"


def pct(val):
    """Format as percentage."""
    return f"{safe_float(val)*100:.0f}%"


# ─────────────────────────────────────────────────────────────
# TABLE GENERATORS
# ─────────────────────────────────────────────────────────────

def generate_dqn_table(rows):
    """Generate DQN hyperparameter table."""
    # Sort by mean reward descending
    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)), reverse=True)
    
    lines = []
    lines.append("### Table 1: DQN Hyperparameter Experiments (Value-Based)")
    lines.append("")
    lines.append("| Run | Learning Rate | Gamma | Buffer Size | Batch Size | Explore Frac | Epsilon Final | Target Update | Net Arch | Mean Reward | Std | Survival | Mean Pop |")
    lines.append("|-----|--------------|-------|-------------|------------|-------------|---------------|---------------|----------|-------------|-----|----------|----------|")
    
    for r in rows_sorted:
        net = r.get("hp_policy_kwargs", "").replace("{'net_arch': ", "").replace("}", "").strip()
        lines.append(
            f"| {r['run_id']} "
            f"| {r['hp_learning_rate']} "
            f"| {r['hp_gamma']} "
            f"| {int(safe_float(r['hp_buffer_size'])):,} "
            f"| {int(safe_float(r['hp_batch_size']))} "
            f"| {r['hp_exploration_fraction']} "
            f"| {r['hp_exploration_final_eps']} "
            f"| {int(safe_float(r['hp_target_update_interval']))} "
            f"| {net} "
            f"| **{fmt(r['eval_mean_reward'])}** "
            f"| {fmt(r['eval_std_reward'])} "
            f"| {pct(r['eval_survival_rate'])} "
            f"| {fmt(r['eval_mean_final_pop'], 3)} |"
        )
    
    # Analysis paragraph
    best = rows_sorted[0]
    worst = rows_sorted[-1]
    
    lines.append("")
    lines.append("**Analysis:**")
    lines.append("")
    
    # Find which hyperparameters mattered most
    survivals = {r['run_id']: safe_float(r['eval_survival_rate']) for r in rows}
    rewards = {r['run_id']: safe_float(r['eval_mean_reward']) for r in rows}
    
    lines.append(
        f"The best DQN configuration was Run {best['run_id']} ({best.get('notes', '').split(chr(8212))[0].split('—')[0].strip()}) "
        f"achieving a mean reward of {fmt(best['eval_mean_reward'])} with "
        f"{pct(best['eval_survival_rate'])} survival rate. "
        f"The worst performer was Run {worst['run_id']} with reward {fmt(worst['eval_mean_reward'])} "
        f"and {pct(worst['eval_survival_rate'])} survival."
    )
    lines.append("")
    
    # Key observations
    lines.append(
        "**Key findings on hyperparameter sensitivity:**"
    )
    lines.append("")
    lines.append(
        f"1. **Gamma (discount factor)** was the most impactful hyperparameter. "
        f"Lower gamma values (0.95-0.97) consistently outperformed high gamma (0.99-0.995) "
        f"because the conservation environment has strong immediate feedback — poaching impacts "
        f"and habitat degradation are felt within a few timesteps. A shorter planning horizon "
        f"helps DQN focus on the most urgent interventions rather than over-discounting future states."
    )
    lines.append("")
    lines.append(
        f"2. **Exploration fraction** significantly affected convergence. "
        f"Run 5 (exploration_fraction=0.5) achieved {pct(rows_sorted[list(r['run_id'] for r in rows_sorted).index('5')]['eval_survival_rate'] if '5' in [r['run_id'] for r in rows_sorted] else 0)} survival, "
        f"suggesting that extended exploration helps DQN discover effective zone-action combinations "
        f"across the 48-action discrete space. Short exploration (0.2) in Run 9 still performed well, "
        f"indicating that the aggressive learning rate compensated."
    )
    lines.append("")
    lines.append(
        f"3. **Network architecture** showed diminishing returns with depth. "
        f"The 3-layer network (Run 7) underperformed the 2-layer baseline, "
        f"likely due to overfitting on the relatively simple 59-dimensional observation space. "
        f"The smaller [128, 128] network (Run 8) also struggled, suggesting insufficient capacity "
        f"for the 48-action output layer."
    )
    lines.append("")
    lines.append(
        f"4. **Buffer size and batch size** had moderate impact. Larger buffers (200K) with "
        f"larger batches (128) improved stability (lower std deviation), indicating that "
        f"diverse experience replay helps DQN generalize across the stochastic climate dynamics."
    )
    
    return "\n".join(lines)


def generate_reinforce_table(rows):
    """Generate REINFORCE hyperparameter table."""
    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)), reverse=True)
    
    lines = []
    lines.append("### Table 2: REINFORCE Hyperparameter Experiments (Policy Gradient)")
    lines.append("")
    lines.append("| Run | Learning Rate | Gamma | Hidden Layers | Entropy Coef | Baseline | Eps/Update | Optimizer | Mean Reward | Std | Survival | Mean Pop |")
    lines.append("|-----|--------------|-------|---------------|-------------|----------|-----------|-----------|-------------|-----|----------|----------|")
    
    for r in rows_sorted:
        lines.append(
            f"| {r['run_id']} "
            f"| {r['hp_learning_rate']} "
            f"| {r['hp_gamma']} "
            f"| {r['hp_hidden_layers']} "
            f"| {r['hp_entropy_coef']} "
            f"| {r['hp_use_baseline']} "
            f"| {r['hp_episodes_per_update']} "
            f"| {r['hp_optimizer']} "
            f"| **{fmt(r['eval_mean_reward'])}** "
            f"| {fmt(r['eval_std_reward'])} "
            f"| {pct(r['eval_survival_rate'])} "
            f"| {fmt(r['eval_mean_final_pop'], 3)} |"
        )
    
    best = rows_sorted[0]
    worst = rows_sorted[-1]
    
    lines.append("")
    lines.append("**Analysis:**")
    lines.append("")
    lines.append(
        f"REINFORCE showed the weakest overall performance among the three algorithms, "
        f"with 0% survival rate across all 10 configurations. The best run was Run {best['run_id']} "
        f"({best.get('notes', '').split(chr(8212))[0].split('—')[0].strip()}) "
        f"with mean reward {fmt(best['eval_mean_reward'])}, while the worst was Run {worst['run_id']} "
        f"at {fmt(worst['eval_mean_reward'])}. This is expected behavior for vanilla REINFORCE — "
        f"its high variance gradient estimates make it difficult to learn stable policies in "
        f"stochastic environments."
    )
    lines.append("")
    lines.append("**Key findings on hyperparameter sensitivity:**")
    lines.append("")
    lines.append(
        f"1. **Baseline subtraction** had a counterintuitive effect. Run 2 (no baseline) "
        f"outperformed Run 1 (with baseline) with reward {fmt(rows[1]['eval_mean_reward'])} vs "
        f"{fmt(rows[0]['eval_mean_reward'])}. This suggests the running average baseline was "
        f"poorly calibrated for the non-stationary reward distribution in our environment, "
        f"where early episodes have fundamentally different reward scales than later ones."
    )
    lines.append("")
    lines.append(
        f"2. **Entropy coefficient** was critical for stability. High entropy (Run 6, coef=0.05) "
        f"produced the worst results ({fmt(rows[5]['eval_mean_reward'])}), indicating excessive "
        f"exploration prevented the policy from exploiting learned strategies. Very low entropy "
        f"(Run 9, coef=0.001) also underperformed, showing premature convergence to suboptimal policies."
    )
    lines.append("")
    lines.append(
        f"3. **Network size** showed that smaller networks (Run 8, [128,64]) performed comparably "
        f"to larger ones, suggesting REINFORCE's bottleneck is the gradient estimation quality, "
        f"not model capacity. The 3-layer network (Run 7) added parameters without benefit."
    )
    lines.append("")
    lines.append(
        f"4. **Episodes per update** had a positive effect when increased. Run 8 (10 episodes/update) "
        f"outperformed Run 1 (5 episodes/update) at the same learning rate, consistent with "
        f"REINFORCE theory — more episodes per batch reduce gradient variance."
    )
    lines.append("")
    lines.append(
        f"5. **Gamma** at 0.95 (Run 5) performed better than 0.99 (Run 1), mirroring the DQN finding "
        f"that shorter planning horizons suit this environment's immediate feedback structure."
    )
    
    return "\n".join(lines)


def generate_ppo_table(rows):
    """Generate PPO hyperparameter table."""
    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)), reverse=True)
    
    lines = []
    lines.append("### Table 3: PPO Hyperparameter Experiments (Policy Gradient)")
    lines.append("")
    lines.append("| Run | Learning Rate | Gamma | Clip Range | Epochs | Batch | N Steps | GAE Lambda | Ent Coef | VF Coef | Mean Reward | Std | Survival | Mean Pop |")
    lines.append("|-----|--------------|-------|-----------|--------|-------|---------|-----------|---------|---------|-------------|-----|----------|----------|")
    
    for r in rows_sorted:
        lines.append(
            f"| {r['run_id']} "
            f"| {r['hp_learning_rate']} "
            f"| {r['hp_gamma']} "
            f"| {r['hp_clip_range']} "
            f"| {int(safe_float(r['hp_n_epochs']))} "
            f"| {int(safe_float(r['hp_batch_size']))} "
            f"| {int(safe_float(r['hp_n_steps']))} "
            f"| {r['hp_gae_lambda']} "
            f"| {r['hp_ent_coef']} "
            f"| {r['hp_vf_coef']} "
            f"| **{fmt(r['eval_mean_reward'])}** "
            f"| {fmt(r['eval_std_reward'])} "
            f"| {pct(r['eval_survival_rate'])} "
            f"| {fmt(r['eval_mean_final_pop'], 3)} |"
        )
    
    best = rows_sorted[0]
    worst = rows_sorted[-1]
    
    lines.append("")
    lines.append("**Analysis:**")
    lines.append("")
    lines.append(
        f"PPO was the strongest algorithm, achieving 100% survival in 9 out of 10 configurations. "
        f"The best configuration was Run {best['run_id']} "
        f"({best.get('notes', '').split(chr(8212))[0].split('—')[0].strip()}) "
        f"with mean reward {fmt(best['eval_mean_reward'])} and mean final population {fmt(best['eval_mean_final_pop'], 3)}. "
        f"Even the worst PPO run (Run {worst['run_id']}, reward {fmt(worst['eval_mean_reward'])}) "
        f"outperformed the best DQN and REINFORCE configurations, demonstrating PPO's robustness "
        f"to hyperparameter choices."
    )
    lines.append("")
    lines.append("**Key findings on hyperparameter sensitivity:**")
    lines.append("")
    lines.append(
        f"1. **Rollout length and batch size** had the largest impact. Run 8 (n_steps=4096, batch=128) "
        f"achieved the highest reward ({fmt(rows_sorted[0]['eval_mean_reward'])}), suggesting that "
        f"longer rollouts capture the multi-step consequences of conservation actions — a patrol "
        f"deployed in month 1 affects poaching pressure for several months, and larger batches "
        f"help the value function estimate these delayed effects."
    )
    lines.append("")
    lines.append(
        f"2. **Clip range** showed that both tight (0.1, Run 4) and wide (0.3, Run 5) clipping "
        f"performed well, with only ~8 points separating them. This indicates PPO's clipping "
        f"mechanism is effective across a range of values for this environment. The default 0.2 "
        f"was near-optimal."
    )
    lines.append("")
    lines.append(
        f"3. **Learning rate** was the most sensitive parameter. The low learning rate (Run 3, 5e-5) "
        f"was the only PPO run with 0% survival (reward {fmt(rows[2]['eval_mean_reward'])}), "
        f"indicating insufficient policy updates within 500K timesteps. The optimal range was "
        f"1e-4 to 3e-4."
    )
    lines.append("")
    lines.append(
        f"4. **Gamma and GAE lambda** together controlled the bias-variance tradeoff. "
        f"Low gamma with low GAE lambda (Run 7, gamma=0.95, lambda=0.9) reduced reward by ~15% "
        f"compared to the baseline (gamma=0.99, lambda=0.95), suggesting that unlike DQN, "
        f"PPO benefits from a longer planning horizon because the value function can learn to "
        f"predict future consequences of current actions."
    )
    lines.append("")
    lines.append(
        f"5. **Entropy coefficient** had minimal impact in the 0.01-0.05 range. "
        f"Run 6 (ent_coef=0.05) performed nearly identically to Run 1 (ent_coef=0.01), "
        f"indicating PPO's natural exploration through stochastic policy sampling was sufficient "
        f"without additional entropy bonuses."
    )
    
    return "\n".join(lines)


def generate_comparison_section(dqn_rows, reinforce_rows, ppo_rows):
    """Generate cross-algorithm comparison."""
    lines = []
    lines.append("### Cross-Algorithm Comparison")
    lines.append("")
    
    # Best from each
    best_dqn = max(dqn_rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)))
    best_reinforce = max(reinforce_rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)))
    best_ppo = max(ppo_rows, key=lambda r: safe_float(r.get("eval_mean_reward", 0)))
    
    lines.append("| Metric | DQN (Best) | REINFORCE (Best) | PPO (Best) |")
    lines.append("|--------|-----------|-----------------|-----------|")
    lines.append(f"| Best Run | Run {best_dqn['run_id']} | Run {best_reinforce['run_id']} | Run {best_ppo['run_id']} |")
    lines.append(f"| Mean Reward | {fmt(best_dqn['eval_mean_reward'])} | {fmt(best_reinforce['eval_mean_reward'])} | **{fmt(best_ppo['eval_mean_reward'])}** |")
    lines.append(f"| Std Reward | {fmt(best_dqn['eval_std_reward'])} | {fmt(best_reinforce['eval_std_reward'])} | {fmt(best_ppo['eval_std_reward'])} |")
    lines.append(f"| Survival Rate | {pct(best_dqn['eval_survival_rate'])} | {pct(best_reinforce['eval_survival_rate'])} | **{pct(best_ppo['eval_survival_rate'])}** |")
    lines.append(f"| Mean Final Pop | {fmt(best_dqn['eval_mean_final_pop'], 3)} | {fmt(best_reinforce['eval_mean_final_pop'], 3)} | **{fmt(best_ppo['eval_mean_final_pop'], 3)}** |")
    lines.append(f"| Mean Final Habitat | {fmt(best_dqn['eval_mean_final_habitat'], 3)} | {fmt(best_reinforce['eval_mean_final_habitat'], 3)} | {fmt(best_ppo['eval_mean_final_habitat'], 3)} |")
    lines.append(f"| Training Time | {fmt(best_dqn['training_time_seconds'])}s | {fmt(best_reinforce['training_time_seconds'])}s | {fmt(best_ppo['training_time_seconds'])}s |")
    
    lines.append("")
    lines.append("**Discussion:**")
    lines.append("")
    lines.append(
        f"PPO demonstrated clear superiority over both DQN and REINFORCE in this conservation "
        f"resource allocation task. PPO's best run achieved {fmt(best_ppo['eval_mean_reward'])} mean reward "
        f"with 100% survival, compared to DQN's {fmt(best_dqn['eval_mean_reward'])} and REINFORCE's "
        f"{fmt(best_reinforce['eval_mean_reward'])}. The ranking PPO > DQN > REINFORCE is consistent "
        f"with the theoretical expectations for this type of environment:"
    )
    lines.append("")
    lines.append(
        f"- **PPO** benefits from its actor-critic architecture, which combines a learned value function "
        f"(reducing variance) with direct policy optimization (maintaining flexibility). The clipped "
        f"surrogate objective prevents destructive policy updates, which is critical in our stochastic "
        f"environment where a single bad update could lead to extinction cascades."
    )
    lines.append("")
    lines.append(
        f"- **DQN** performed second-best because value-based methods naturally handle the discrete "
        f"action space (48 actions), but struggled with the sequential nature of conservation decisions "
        f"where the optimal action depends heavily on which zones were recently serviced."
    )
    lines.append("")
    lines.append(
        f"- **REINFORCE** performed worst due to its inherent high-variance gradient estimates. "
        f"With only Monte Carlo returns and no value baseline (or a poorly calibrated one), "
        f"the policy gradient signal was too noisy to consistently improve in an environment "
        f"with stochastic climate events and 48 possible actions per step."
    )
    lines.append("")
    lines.append(
        f"The most important insight across all algorithms was that **shorter discount horizons "
        f"(lower gamma) helped DQN but hurt PPO**. This reflects a fundamental difference: "
        f"DQN's value function directly estimates discounted returns and benefits from focusing "
        f"on immediate rewards, while PPO's advantage estimation (via GAE) needs longer horizons "
        f"to accurately assess the multi-step impact of conservation interventions."
    )
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# HTML WRAPPER
# ─────────────────────────────────────────────────────────────

def markdown_to_html(md_content):
    """Simple markdown to HTML conversion for tables."""
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hyperparameter Experiment Tables</title>
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }
h3 { color: #1a5276; border-bottom: 2px solid #2196F3; padding-bottom: 8px; margin-top: 40px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; }
th { background: #f0f4f8; padding: 8px 10px; text-align: left; border: 1px solid #ddd; font-weight: 600; }
td { padding: 6px 10px; border: 1px solid #ddd; }
tr:nth-child(even) { background: #fafafa; }
tr:first-child td { font-weight: bold; background: #e8f5e9; }
strong { color: #1a5276; }
p { margin: 8px 0; }
</style>
</head>
<body>
<h1>Hyperparameter Experiment Tables</h1>
"""
    
    lines = md_content.split("\n")
    in_table = False
    
    for line in lines:
        if line.startswith("### "):
            if in_table:
                html += "</table>\n"
                in_table = False
            html += f"<h3>{line[4:]}</h3>\n"
        elif line.startswith("|") and "---" in line:
            continue  # skip separator
        elif line.startswith("|"):
            if not in_table:
                html += "<table>\n"
                in_table = True
                cells = [c.strip() for c in line.split("|")[1:-1]]
                html += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>\n"
            else:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>\n"
        else:
            if in_table:
                html += "</table>\n"
                in_table = False
            if line.startswith("**") and line.endswith("**"):
                html += f"<p><strong>{line[2:-2]}</strong></p>\n"
            elif line.strip():
                html += f"<p>{line}</p>\n"
    
    if in_table:
        html += "</table>\n"
    
    html += "</body></html>"
    return html


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating Hyperparameter Tables for Report")
    print("=" * 60)
    
    dqn_rows = read_csv(os.path.join(RESULTS_DIR, "dqn", "dqn_experiments.csv"))
    reinforce_rows = read_csv(os.path.join(RESULTS_DIR, "reinforce", "reinforce_experiments.csv"))
    ppo_rows = read_csv(os.path.join(RESULTS_DIR, "ppo", "ppo_experiments.csv"))
    
    print(f"  DQN: {len(dqn_rows)} runs")
    print(f"  REINFORCE: {len(reinforce_rows)} runs")
    print(f"  PPO: {len(ppo_rows)} runs")
    
    # Generate markdown
    sections = []
    
    if dqn_rows:
        sections.append(generate_dqn_table(dqn_rows))
    if reinforce_rows:
        sections.append(generate_reinforce_table(reinforce_rows))
    if ppo_rows:
        sections.append(generate_ppo_table(ppo_rows))
    if dqn_rows and reinforce_rows and ppo_rows:
        sections.append(generate_comparison_section(dqn_rows, reinforce_rows, ppo_rows))
    
    md_content = "\n\n---\n\n".join(sections)
    
    # Save markdown
    md_path = os.path.join(TABLES_DIR, "hyperparameter_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Saved: {md_path}")
    
    # Save HTML preview
    html_content = markdown_to_html(md_content)
    html_path = os.path.join(TABLES_DIR, "hyperparameter_tables.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Saved: {html_path}")
    
    print(f"\n  Open {html_path} in a browser for visual preview")
    print("  Copy content from {md_path} into your report")
    print("=" * 60)


if __name__ == "__main__":
    main()