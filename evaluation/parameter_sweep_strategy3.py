#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import matplotlib.pyplot as plt

def main():
    configs = [
        {"id": 1, "temp": 1.0, "top_p": 0.95, "top_k": 50, "desc": "Default Baseline"},
        {"id": 2, "temp": 0.2, "top_p": 0.95, "top_k": 50, "desc": "Low Temp (Deterministic)"},
        {"id": 3, "temp": 0.5, "top_p": 0.95, "top_k": 50, "desc": "Conservative"},
        {"id": 4, "temp": 0.7, "top_p": 0.90, "top_k": 50, "desc": "Balanced-Low"},
        {"id": 5, "temp": 1.0, "top_p": 0.80, "top_k": 50, "desc": "Nucleus-restricted"},
        {"id": 6, "temp": 1.2, "top_p": 0.95, "top_k": 50, "desc": "High Temp (Diverse)"},
        {"id": 7, "temp": 0.7, "top_p": 0.90, "top_k": 20, "desc": "Restricted Pool"}
    ]

    base_dir = "runs/sweep_strategy3"
    os.makedirs(base_dir, exist_ok=True)
    results = []

    print("=" * 80)
    print("Starting Hyperparameter Sweep for Strategy 3 (baseline_adapter_softmax)")
    print("=" * 80)

    for cfg in configs:
        cfg_id = cfg["id"]
        temp = cfg["temp"]
        top_p = cfg["top_p"]
        top_k = cfg["top_k"]
        desc = cfg["desc"]
        out_dir = os.path.join(base_dir, f"config_{cfg_id}")
        
        print(f"\n[Config {cfg_id}/7] Running: Temp={temp}, Top-p={top_p}, Top-k={top_k} ({desc})")
        
        cmd = [
            sys.executable, "evaluation/benchmark_gsi_strategies_harmlessness.py",
            "--strategies", "baseline_adapter_softmax",
            "--num-prompts", "15",
            "--max-tokens", "400",
            "--temperature", str(temp),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--output-dir", out_dir
        ]
        
        # Run the benchmark script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running config {cfg_id}: {e}")
            continue
            
        # Parse results
        results_file = os.path.join(out_dir, "baseline_adapter_softmax_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
                results.append({
                    "id": cfg_id,
                    "temp": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "desc": desc,
                    "avg_blade_reward": data.get("avg_blade_reward"),
                    "std_blade_reward": data.get("std_blade_reward"),
                    "refusal_rate": data.get("refusal_rate"),
                    "elapsed_s": data.get("elapsed_s")
                })
        else:
            print(f"Results file not found at {results_file}")

    # Display comparison table
    print("\n" + "=" * 90)
    print("  Strategy 3 Hyperparameter Sweep Results")
    print("=" * 90)
    print(f"  {'ID':<3} {'Temp':<6} {'Top-p':<6} {'Top-k':<6} {'AvgReward':>12} {'Std':>8} {'Refusal%':>9} {'Time':>8}  {'Description'}")
    print(f"  {'─' * 3} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 12} {'─' * 8} {'─' * 9} {'─' * 8}  {'─' * 30}")

    for r in results:
        refusal_pct = r['refusal_rate'] * 100 if r['refusal_rate'] is not None else 0.0
        print(
            f"  {r['id']:<3} {r['temp']:<6.1f} {r['top_p']:<6.2f} {r['top_k']:<6} "
            f"{r['avg_blade_reward']:>12.5f} {r['std_blade_reward']:>8.4f} "
            f"{refusal_pct:>8.1f}% {r['elapsed_s']:>7.1f}s  {r['desc']}"
        )
    print("=" * 90)

    # Save summary json
    summary_path = os.path.join(base_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary JSON to {summary_path}")

    # Generate Comparison Plot
    if results:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Config ID / Description')
        ax1.set_ylabel('Avg Blade Reward (Harmlessness)', color=color)
        labels = [f"C{r['id']}\n({r['desc']})" for r in results]
        x = range(len(results))
        ax1.bar(x, [r['avg_blade_reward'] for r in results], color=color, alpha=0.6, width=0.4, label='Avg Blade Reward')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Refusal Rate (%)', color=color)
        ax2.plot(x, [r['refusal_rate']*100 for r in results], color=color, marker='o', linewidth=2, label='Refusal Rate')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Strategy 3 Hyperparameter Sweep: Reward vs Refusal Rate')
        plot_path = os.path.join(base_dir, "sweep_comparison.png")
        plt.savefig(plot_path)
        print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
