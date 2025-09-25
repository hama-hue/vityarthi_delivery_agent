"""
src/experiments.py

Runs systematic experiments with UCS, A*, and Local Search planners.
Collects metrics (cost, nodes expanded, time, collisions) and saves to CSV.

Usage:
    python src/experiments.py --maps maps/map_small.txt maps/map_dynamic.txt --schedule maps/dynamic_schedule.json --out results.csv

Notes:
- UCS is included mainly for comparison; may be slow on large maps.
- A* and LocalSearch are the main planners.
"""

import argparse
import csv
import os
import time
from typing import List, Dict, Any

from grid import Grid
from search import uniform_cost_search, a_star_search
from local_search import LocalSearchPlanner
from agent import Agent, Simulator


def run_experiment_on_map(map_file: str,
                          schedule_file: str = None,
                          max_steps: int = 200,
                          seed: int = 42) -> List[Dict[str, Any]]:
    """Run UCS, A*, LocalSearch on given map. Returns list of result dicts."""
    results = []
    g = Grid.from_files(map_file, schedule_file)

    # --- UCS (spatial only) ---
    t0 = time.perf_counter()
    path_u, cost_u, nodes_u, ms_u = uniform_cost_search(g)
    elapsed_u = (time.perf_counter() - t0) * 1000.0
    results.append({
        "map": os.path.basename(map_file),
        "planner": "UCS",
        "path_found": path_u is not None,
        "path_cost": cost_u,
        "nodes_expanded": nodes_u,
        "planning_time_ms": ms_u,
        "sim_steps": None,
        "collisions": None
    })

    # --- A* Agent + Simulation ---
    agent_a = Agent(g, planner_type="astar")
    sim_a = Simulator(g, agent_a, max_steps=max_steps)
    res_a = sim_a.run(verbose=False)
    results.append({
        "map": os.path.basename(map_file),
        "planner": "A*",
        "path_found": res_a["finished"],
        "path_cost": None,  # cost not directly tracked in agent; could recompute from path_taken
        "nodes_expanded": None,
        "planning_time_ms": None,
        "sim_steps": res_a["steps_taken"],
        "collisions": res_a["collisions"]
    })

    # --- Local Search Agent + Simulation ---
    agent_l = Agent(g, planner_type="local", seed=seed)
    sim_l = Simulator(g, agent_l, max_steps=max_steps)
    res_l = sim_l.run(verbose=False)
    results.append({
        "map": os.path.basename(map_file),
        "planner": "LocalSearch",
        "path_found": res_l["finished"],
        "path_cost": None,
        "nodes_expanded": None,
        "planning_time_ms": None,
        "sim_steps": res_l["steps_taken"],
        "collisions": res_l["collisions"]
    })

    return results


def save_results_csv(results: List[Dict[str, Any]], out_file: str):
    """Save list of result dicts to CSV."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments on maps")
    parser.add_argument("--maps", nargs="+", required=True, help="List of map files")
    parser.add_argument("--schedule", help="Schedule file (if any)")
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    parser.add_argument("--steps", type=int, default=200, help="Max simulation steps")
    args = parser.parse_args()

    all_results = []
    for map_file in args.maps:
        print(f"\n=== Running experiments on {map_file} ===")
        res = run_experiment_on_map(map_file, args.schedule, max_steps=args.steps)
        all_results.extend(res)

    save_results_csv(all_results, args.out)


if __name__ == "__main__":
    main()

