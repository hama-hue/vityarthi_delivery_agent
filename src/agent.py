"""
src/agent.py

Implements:
- Agent: entity that moves in the grid, follows a planned path, can replan
- Simulator: runs a simulation loop, advancing time, moving the agent, detecting collisions

Usage:
    from agent import Agent, Simulator
    from grid import Grid
    from search import a_star_search
    from local_search import LocalSearchPlanner
"""

from typing import List, Tuple, Optional, Dict, Any
import time

from grid import Grid
from search import a_star_search
from local_search import LocalSearchPlanner

Position = Tuple[int, int]


class Agent:
    def __init__(self,
                 grid: Grid,
                 start: Optional[Position] = None,
                 goal: Optional[Position] = None,
                 planner_type: str = "astar",
                 seed: Optional[int] = None):
        """
        planner_type: "astar" or "local"
        """
        self.grid = grid
        self.start = start if start is not None else grid.start
        self.goal = goal if goal is not None else grid.goal
        self.pos = self.start
        self.time = 0
        self.path: List[Position] = []
        self.planner_type = planner_type
        self.local_planner = None
        if planner_type == "local":
            self.local_planner = LocalSearchPlanner(grid, seed=seed)
        self.finished = False
        self.collisions = 0

    def plan(self) -> List[Position]:
        """Plan a path from current pos to goal using selected planner."""
        if self.planner_type == "astar":
            path, cost, nodes, ms = a_star_search(self.grid, start=self.pos, goal=self.goal)
            self.path = path if path else []
        elif self.planner_type == "local":
            path, score, stats = self.local_planner.plan(start=self.pos, goal=self.goal, current_time=self.time)
            self.path = path if path else []
        else:
            raise ValueError("Unknown planner_type: " + str(self.planner_type))
        return self.path

    def step(self):
        """Advance the agent one timestep along its path. Handle collisions if they occur."""
        if self.finished:
            return

        # If no path, try to replan
        if not self.path or self.pos == self.path[-1]:
            if self.pos == self.goal:
                self.finished = True
                return
            self.plan()
            if not self.path:
                # Stuck: no path found
                self.finished = True
                return

        # Next position = the next element in path
        # Our path includes current pos, so drop it if needed
        if self.path and self.path[0] == self.pos:
            self.path.pop(0)

        if not self.path:
            # Already at goal
            self.finished = True
            return

        next_pos = self.path[0]

        # Move agent
        self.time += 1
        if self.grid.passable(next_pos, self.time):
            self.pos = next_pos
        else:
            # Collision with dynamic obstacle
            self.collisions += 1
            # Stay in place this turn
            # Optionally trigger replanning next time
        # Do not pop next_pos until actually moved
        if self.pos == next_pos:
            self.path.pop(0)

        if self.pos == self.goal:
            self.finished = True


class Simulator:
    def __init__(self, grid: Grid, agent: Agent, max_steps: int = 500):
        self.grid = grid
        self.agent = agent
        self.max_steps = max_steps
        self.log: List[Dict[str, Any]] = []

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run until agent finishes or max_steps exceeded. Logs every step."""
        t0 = time.perf_counter()
        for step in range(self.max_steps):
            if self.agent.finished:
                break
            self.agent.step()
            self.log.append({
                "time": self.agent.time,
                "pos": self.agent.pos,
                "collisions": self.agent.collisions
            })
            if verbose:
                print(f"t={self.agent.time}, pos={self.agent.pos}, collisions={self.agent.collisions}")
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = {
            "finished": self.agent.finished,
            "time": self.agent.time,
            "steps_taken": len(self.log),
            "collisions": self.agent.collisions,
            "elapsed_ms": elapsed,
            "path_taken": [entry["pos"] for entry in self.log]
        }
        if verbose:
            print("\nSimulation finished:", result)
        return result


# -----------------------
# Demo when run directly
# -----------------------
if __name__ == "__main__":
    print("Agent & Simulator demo on ../maps/map_dynamic.txt")
    g = Grid.from_files("../maps/map_dynamic.txt", "../maps/dynamic_schedule.json")

    # A* agent
    print("\n=== Running A* Agent ===")
    agent1 = Agent(g, planner_type="astar")
    sim1 = Simulator(g, agent1, max_steps=50)
    res1 = sim1.run(verbose=True)

    # LocalSearch agent
    print("\n=== Running LocalSearch Agent ===")
    agent2 = Agent(g, planner_type="local", seed=123)
    sim2 = Simulator(g, agent2, max_steps=50)
    res2 = sim2.run(verbose=True)

