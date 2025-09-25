"""
src/visualize.py

Visualization utilities for the Vityarthi delivery agent project.

Functions:
- draw_grid(ax, grid): draw terrain + blocked cells
- draw_agent_and_path(ax, agent_pos, path_history, planned_path): overlay the agent and paths
- draw_obstacles(ax, grid, t): draw moving obstacles at time t (uses grid.schedule)
- render_simulation(grid, sim_log, frames_dir="frames", show_planned=True, planned_path=None)
    - sim_log: list of dicts like Simulator.log [{'time': t, 'pos': (r,c), 'collisions': n}, ...]
    - saves frames to frames_dir/frame_0000.png ...
- Example: run `python src/visualize.py` from project root to run a demo on maps/map_dynamic.txt

Notes:
- This script uses matplotlib. Install with: pip install matplotlib numpy
- To convert frames to mp4 (ffmpeg must be installed):
    ffmpeg -framerate 8 -i frames/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p demo.mp4
"""

import os
import shutil
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

# local imports (works when run as: python src/visualize.py from project root)
from grid import Grid
from agent import Simulator, Agent


Position = Tuple[int, int]


# -------------------------
# Drawing helpers
# -------------------------
def draw_grid(ax, grid: Grid, show_coords: bool = False):
    """Draw the grid: terrain costs and static obstacles."""
    rows, cols = grid.rows, grid.cols

    # collect cost range
    costs = [grid.grid[r][c] for r in range(rows) for c in range(cols) if grid.grid[r][c] is not None]
    min_cost = min(costs) if costs else 1
    max_cost = max(costs) if costs else 1
    denom = max(1e-6, (max_cost - min_cost))

    cmap = cm.get_cmap("viridis")

    # Draw each cell as rectangle
    for r in range(rows):
        for c in range(cols):
            val = grid.grid[r][c]
            if val is None:
                face = (0.0, 0.0, 0.0)  # black for blocked
            else:
                norm = (float(val) - min_cost) / denom
                face = cmap(norm)[:3]
            rect = patches.Rectangle((c, r), 1.0, 1.0, linewidth=0.5, edgecolor="gray", facecolor=face)
            ax.add_patch(rect)

    # grid lines and axes
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)  # invert y so row=0 is top
    ax.set_aspect("equal")
    ax.set_xticks([x for x in range(cols + 1)])
    ax.set_yticks([y for y in range(rows + 1)])
    ax.grid(False)
    if not show_coords:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def draw_start_goal(ax, grid: Grid):
    """Plot start and goal markers (centered in the cell)."""
    sx, sy = grid.start[1] + 0.5, grid.start[0] + 0.5
    gx, gy = grid.goal[1] + 0.5, grid.goal[0] + 0.5
    ax.scatter([sx], [sy], marker="*", s=180, c="white", edgecolors="k", linewidths=1.2, label="Start")
    ax.scatter([gx], [gy], marker="X", s=140, c="white", edgecolors="k", linewidths=1.2, label="Goal")


def draw_agent_and_path(ax, agent_pos: Position, path_history: List[Position], planned_path: Optional[List[Position]] = None):
    """Plot agent history (solid), current pos (circle), and planned path (dashed)."""
    if path_history:
        xs = [p[1] + 0.5 for p in path_history]
        ys = [p[0] + 0.5 for p in path_history]
        ax.plot(xs, ys, linestyle="-", linewidth=2.0, marker="o", markersize=4, label="Path taken")

    # planned path (if provided)
    if planned_path:
        xs = [p[1] + 0.5 for p in planned_path]
        ys = [p[0] + 0.5 for p in planned_path]
        ax.plot(xs, ys, linestyle="--", linewidth=1.6, marker=".", markersize=6, alpha=0.9, label="Planned path")

    # current agent position
    if agent_pos is not None:
        ax.scatter([agent_pos[1] + 0.5], [agent_pos[0] + 0.5], s=120, c="red", marker="o", edgecolors="k", label="Agent")


def obstacle_positions_by_id(grid: Grid, t: int) -> Dict[str, Position]:
    """Return dict: obstacle_id -> pos at time t (if present)"""
    res = {}
    for obs_id, traj in grid.schedule.items():
        if not isinstance(traj, dict):
            # some schedules may be stored as mapping with string keys
            try:
                traj = {int(k): tuple(v) for k, v in traj.items()}
            except Exception:
                pass
        pos = traj.get(int(t))
        if pos is not None:
            res[obs_id] = tuple(pos)
    return res


def draw_obstacles(ax, grid: Grid, t: int):
    """Draw moving obstacles present at time t. Each obstacle id gets its own color."""
    obs = obstacle_positions_by_id(grid, t)
    if not obs:
        return

    # color palette
    palette = cm.get_cmap("tab10")
    ids = list(sorted(obs.keys()))
    id_to_color = {oid: palette(i % 10) for i, oid in enumerate(ids)}

    for oid, pos in obs.items():
        r, c = pos
        # draw a small square slightly smaller than the cell
        rect = patches.Rectangle((c + 0.12, r + 0.12), 0.76, 0.76, linewidth=0.6,
                                 edgecolor="k", facecolor=id_to_color[oid], label=f"obs_{oid}")
        ax.add_patch(rect)
        # label
        ax.text(c + 0.5, r + 0.5, str(oid), ha="center", va="center", fontsize=7, color="k")


# -------------------------
# Rendering / frame saving
# -------------------------
def render_simulation(grid: Grid,
                      sim_log: List[Dict],
                      frames_dir: str = "frames",
                      show_planned: bool = True,
                      planned_path: Optional[List[Position]] = None,
                      figsize: Tuple[int, int] = (8, 8),
                      overwrite: bool = True) -> List[str]:
    """
    Render simulation frames from sim_log and save PNG files to frames_dir.
    Returns list of saved frame paths.

    sim_log entries must be dicts with keys: 'time' (int), 'pos' (r,c), optionally 'planned' (list)
    planned_path param can be used to overlay a static planned path for all frames (if desired).
    """
    # prepare frames directory
    if overwrite and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    saved = []
    for idx, entry in enumerate(sim_log):
        t = entry.get("time", idx)
        pos = entry.get("pos")
        # allow per-entry planned path override (entry may include 'planned')
        entry_planned = entry.get("planned", None)
        fig, ax = plt.subplots(figsize=figsize)
        draw_grid(ax, grid)
        draw_start_goal(ax, grid)

        # obstacles at time t
        draw_obstacles(ax, grid, t)

        # path history: all positions up to current idx
        history = [e["pos"] for e in sim_log[: idx + 1]]
        draw_agent_and_path(ax, agent_pos=pos, path_history=history,
                            planned_path=entry_planned if entry_planned is not None else planned_path)

        # timestamp
        ax.text(0.02, 0.02, f"t = {t}", transform=ax.transAxes, fontsize=12, color="white",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))

        # Legend: avoid duplicate labels; create custom legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # dedupe legend entries preserving order
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

        fname = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        saved.append(fname)

    print(f"Saved {len(saved)} frames to {frames_dir}")
    return saved


# -------------------------
# Convenience helpers
# -------------------------
def render_simulator_to_frames(grid: Grid, agent: Agent, frames_dir: str = "frames", max_steps: int = 500):
    """Run a Simulator on the given agent and render frames for the produced log."""
    sim = Simulator(grid, agent, max_steps=max_steps)
    sim_log = sim.run(verbose=False)  # The Simulator.run used earlier returned a dict when finishing.
    # Our Simulator.run in agent.py returns a dict result; its 'path_taken' is list of pos, but not detailed log.
    # To make visualization consistent, we will re-run Simulator with verbose=False but capturing internal log:
    # For compatibility, if sim.run returns dict with 'path_taken' alone, create a simple sim_log.
    if isinstance(sim_log, dict) and "path_taken" in sim_log:
        # create per-step entries
        path = sim_log["path_taken"]
        sim_entries = []
        for i, p in enumerate(path):
            sim_entries.append({"time": i + 1, "pos": p})
    else:
        # assume sim_log is list of dict entries
        sim_entries = sim_log

    return render_simulation(grid, sim_entries, frames_dir=frames_dir, show_planned=False, planned_path=None)


# -------------------------
# Demo when run directly
# -------------------------
if __name__ == "__main__":
    # Demo: run simulator for astar agent and local agent and render frames for the astar run
    print("Visualization demo on ../maps/map_dynamic.txt")
    grid = Grid.from_files("../maps/map_dynamic.txt", "../maps/dynamic_schedule.json")

    # create an A* agent, run simulation and render frames
    agent = Agent(grid, planner_type="astar")
    sim = Simulator(grid, agent, max_steps=80)
    # Run but capture simulator.log from agent.Simulator (the object has attribute `log` after running)
    res = sim.run(verbose=False)
    # The Simulator implementation appends to sim.log before printing; we can use sim.log
    sim_log = sim.log
    # Render frames
    frames = render_simulation(grid, sim_log, frames_dir="frames_astar", planned_path=agent.path, overwrite=True)
    print("Frames created:", len(frames))
    print("To make mp4, run (requires ffmpeg):")
    print("ffmpeg -framerate 8 -i frames_astar/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p demo_astar.mp4")

