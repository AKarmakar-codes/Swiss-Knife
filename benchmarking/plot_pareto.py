"""
Pareto Frontier Plotting Script
===============================

This script reads the tribunal evaluation results from `tribunal/eval_results/`
and generates Pareto frontier plots comparing the different baseline strategies.

It specifically looks for the `model_summary.csv` or `summary.csv` files generated
by the tribunal pipeline. It extracts the Harmlessness score (Y-axis, representing 
alignment safety) and the Response Quality or Helpfulness score (X-axis, representing 
generation capability).

Usage:
    python benchmarking/plot_pareto.py --task harmlessness
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    p = argparse.ArgumentParser(description="Plot Pareto Frontier from Tribunal Results")
    p.add_argument("--task", type=str, default="harmlessness")
    p.add_argument("--output-dir", type=str, default="benchmarking/plots")
    return p.parse_args()


def extract_alpha_from_model_name(model_name, prefix):
    """Extract the hyperparameter value from the file/model name."""
    try:
        # e.g., args_alpha_1.5 -> 1.5
        # e.g., mod_w_helpful_0.6 -> 0.6
        if model_name.startswith(prefix):
            parts = model_name.split("_")
            return float(parts[-1])
        return None
    except ValueError:
        return None


def plot_pareto(df, x_metric, y_metric, output_path):
    """Plot a Pareto frontier comparing different strategies."""
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    strategies = {
        "args": "ARGS (Token Steering)",
        "deal": "DeAL (Top-k Reranking)",
        "mod": "MOD (Linear Mixture)",
    }
    
    colors = sns.color_palette("husl", len(strategies))
    
    for i, (strat_prefix, strat_label) in enumerate(strategies.items()):
        # Filter dataframe for this strategy
        strat_df = df[df['model'].str.startswith(strat_prefix)].copy()
        
        if strat_df.empty:
            continue
            
        # Extract the hyperparameter for sorting/labeling
        strat_df['param'] = strat_df['model'].apply(lambda m: extract_alpha_from_model_name(m, strat_prefix))
        strat_df = strat_df.sort_values(by='param')
        
        # Plot the curve
        plt.plot(
            strat_df[x_metric], 
            strat_df[y_metric], 
            marker='o', 
            markersize=8,
            linewidth=2,
            label=strat_label,
            color=colors[i]
        )
        
        # Annotate points with their hyperparameter value
        for _, row in strat_df.iterrows():
            plt.annotate(
                f"{row['param']:.1f}",
                (row[x_metric], row[y_metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )

    plt.title(f"Pareto Frontier: {y_metric} vs {x_metric}", fontsize=14, pad=15)
    plt.xlabel(x_metric.replace("_score", "").replace("_mean", "").title(), fontsize=12)
    plt.ylabel(y_metric.replace("_score", "").replace("_mean", "").title(), fontsize=12)
    
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto plot to {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    results_dir = os.path.join(os.path.dirname(__file__), "..", "tribunal", "eval_results", args.task)
    summary_file = os.path.join(results_dir, "summary.csv")
    
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found at {summary_file}.")
        print("Please run the hyperparameter sweeps and tribunal evaluation first.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading tribunal results from {summary_file}...")
    df = pd.read_csv(summary_file)
    
    # summary.csv is likely in long format: model, metric, group, n_judged, mean, median, std
    # We need to pivot it to wide format
    if 'metric' in df.columns and 'mean' in df.columns:
        wide_df = df.pivot(index='model', columns='metric', values='mean').reset_index()
    else:
        # If it's already wide format (like model_summary.csv might be)
        wide_df = df
        
    print(f"Found data for {len(wide_df)} model configurations.")
    
    # Identify available metrics
    available_metrics = wide_df.columns.tolist()
    available_metrics.remove('model')
    print(f"Available metrics: {available_metrics}")
    
    # Define primary axes for Pareto
    x_metric_candidates = ['response_quality_score', 'helpfulness_score', 'response_quality']
    y_metric_candidates = ['harmlessness_score', 'harmlessness']
    
    x_metric = next((m for m in x_metric_candidates if m in available_metrics), None)
    y_metric = next((m for m in y_metric_candidates if m in available_metrics), None)
    
    if x_metric and y_metric:
        out_path = os.path.join(args.output_dir, f"pareto_frontier_{args.task}.png")
        plot_pareto(wide_df, x_metric, y_metric, out_path)
    else:
        print(f"Could not find required metrics for Pareto plot.")
        print(f"Need one of {x_metric_candidates} and one of {y_metric_candidates}.")


if __name__ == "__main__":
    main()
