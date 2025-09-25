"""
src/plot_results.py

Reads experiment results CSV (from experiments.py) and plots comparisons.

Usage (from project root):
    python src/plot_results.py --results results.csv --outdir plots

This will create PNG plots for:
- Steps taken vs planner
- Collisions vs planner
- Planning time (if available)
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_steps(df, outdir: str):
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="planner", y="sim_steps", hue="map")
    plt.ylabel("Simulation Steps")
    plt.title("Steps taken vs Planner")
    plt.legend(title="Map")
    fname = os.path.join(outdir, "steps_vs_planner.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def plot_collisions(df, outdir: str):
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="planner", y="collisions", hue="map")
    plt.ylabel("Collisions")
    plt.title("Collisions vs Planner")
    plt.legend(title="Map")
    fname = os.path.join(outdir, "collisions_vs_planner.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def plot_planning_time(df, outdir: str):
    if "planning_time_ms" not in df.columns:
        return
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="planner", y="planning_time_ms", hue="map")
    plt.ylabel("Planning Time (ms)")
    plt.title("Planning Time vs Planner")
    plt.legend(title="Map")
    fname = os.path.join(outdir, "planning_time_vs_planner.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--results", required=True, help="CSV results file from experiments.py")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.results)

    # Remove rows where sim_steps or collisions are NaN (like UCS rows)
    df_steps = df.dropna(subset=["sim_steps"])
    df_coll = df.dropna(subset=["collisions"])
    df_time = df.dropna(subset=["planning_time_ms"])

    if not df_steps.empty:
        plot_steps(df_steps, args.outdir)
    if not df_coll.empty:
        plot_collisions(df_coll, args.outdir)
    if not df_time.empty:
        plot_planning_time(df_time, args.outdir)


if __name__ == "__main__":
    main()
