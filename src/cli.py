"""
src/cli.py

Command-line interface for the Vityarthi delivery agent project.

Usage (from project root):
    python src/cli.py --map maps/map_dynamic.txt --schedule maps/dynamic_schedule.json --planner astar --steps 80 --visualize

Arguments:
    --map <path>        Path to map file (required)
    --schedule <path>   Path to dynamic obstacle schedule (optional)
    --planner <name>    Planner type: "astar" or "local" (default: astar)
    --steps <int>       Max simulation steps (default: 100)
    --visualize         If set, render simulation frames
    --frames <dir>      Directory to save frames (default: frames_cli)

Example:
    python src/cli.py --map maps/map_small.txt --planner local --steps 50 --visualize
"""

import argparse
import sys

from grid import Grid
from agent import Agent, Simulator
from visualize import render_simulation


def main():
    parser = argparse.ArgumentParser(description="Vityarthi Delivery Agent CLI")
    parser.add_argument("--map", required=True, help="Path to map file (txt)")
    parser.add_argument("--schedule", help="Path to dynamic obstacle schedule (json)")
    parser.add_argument("--planner", default="astar", choices=["astar", "local"], help="Planner type")
    parser.add_argument("--steps", type=int, default=100, help="Max simulation steps")
    parser.add_argument("--visualize", action="store_true", help="Render visualization frames")
    parser.add_argument("--frames", default="frames_cli", help="Output directory for frames (if visualize)")

    args = parser.parse_args()

    # Load grid
    print(f"Loading grid from {args.map} ...")
    g = Grid.from_files(args.map, args.schedule)

    # Create agent and simulator
    print(f"Creating agent with planner={args.planner}")
    agent = Agent(g, planner_type=args.planner)
    sim = Simulator(g, agent, max_steps=args.steps)

    # Run simulation
    print("Running simulation...")
    result = sim.run(verbose=True)

    print("\n=== Simulation Summary ===")
    print("Finished:", result["finished"])
    print("Steps taken:", result["steps_taken"])
    print("Time elapsed:", result["time"], "timesteps")
    print("Collisions:", result["collisions"])
    print("Path length:", len(result["path_taken"]))

    # Visualization
    if args.visualize:
        print(f"\nRendering frames to {args.frames} ...")
        render_simulation(g, sim.log, frames_dir=args.frames, planned_path=agent.path, overwrite=True)
        print("Frames ready. To convert to mp4 (requires ffmpeg):")
        print(f"ffmpeg -framerate 8 -i {args.frames}/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p {args.frames}.mp4")


if __name__ == "__main__":
    main()

