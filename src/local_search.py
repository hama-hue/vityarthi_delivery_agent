"""
src/local_search.py

Local search planner: Hill-climbing with random restarts for replanning / quick
path improvements on grid maps.

API:
    planner = LocalSearchPlanner(grid, start=None, goal=None, seed=42)
    best_path, best_score, stats = planner.plan(
        start=None,            # optional override
        goal=None,             # optional override
        current_time=0,        # time at which the agent is at `start`
        max_restarts=5,
        max_iters_per_restart=200,
        patience=40,
        neighbors_per_iter=8,
        collision_penalty=1e6,
        time_limit=None        # seconds
    )

Notes:
- The returned path is a list of (r,c) positions (spatial path).
- Score is sum of cell costs for path (consistent with a_star/search.py)
  plus collision_penalty * (# collisions with moving obstacles when executed
  from current_time).
"""
from typing import List, Tuple, Optional, Dict, Any
import time
import random
import math

from grid import Grid
from search import a_star_search

Position = Tuple[int, int]


class LocalSearchPlanner:
    def __init__(self,
                 grid: Grid,
                 start: Optional[Position] = None,
                 goal: Optional[Position] = None,
                 seed: Optional[int] = None):
        self.grid = grid
        self.start = start if start is not None else grid.start
        self.goal = goal if goal is not None else grid.goal
        self.rng = random.Random(seed)

    # -----------------------
    # Utilities
    # -----------------------
    def path_cost(self, path: List[Position]) -> float:
        """Return path cost consistent with uniform_cost_search / a_star_search.
        cost = sum of cell_cost for every node except the start (cost to *enter* each node).
        """
        if not path:
            return float("inf")
        # If path has n nodes, cost is sum of grid.cell_cost(path[i]) for i=1..n-1
        total = 0.0
        for idx in range(1, len(path)):
            try:
                total += self.grid.cell_cost(path[idx])
            except Exception:
                # blocked or invalid
                return float("inf")
        return total

    def simulate_collisions(self, path: List[Position], start_time: int = 0) -> int:
        """Simulate executing the path from time=start_time and count collisions
        with dynamic obstacles (grid.passable(pos, t) == False).
        Each step (including being at the start) is considered at successive times:
            path[0] at t=start_time, path[1] at t=start_time+1, ...
        """
        collisions = 0
        t = int(start_time)
        for pos in path:
            if not self.grid.passable(pos, t):
                collisions += 1
            t += 1
        return collisions

    def score(self, path: List[Position], start_time: int = 0, collision_penalty: float = 1e6) -> float:
        """Score = path_cost + collision_penalty * collisions_count"""
        c = self.path_cost(path)
        collisions = self.simulate_collisions(path, start_time)
        return c + collision_penalty * collisions

    # -----------------------
    # Path generators / repair
    # -----------------------
    def generate_initial_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """Try A* first; if it fails, attempt a greedy-random walk to find any feasible path."""
        path, cost, nodes_exp, tms = a_star_search(self.grid, start=start, goal=goal, time_aware=False)
        if path:
            return path
        # fallback: greedy-random walk
        return self.greedy_random_walk(start, goal, max_steps=self.grid.rows * self.grid.cols * 4)

    def greedy_random_walk(self, start: Position, goal: Position, max_steps: int = 1000) -> Optional[List[Position]]:
        """Simple random-biased walk that prefers neighbors closer to goal but
        avoids static obstacles. Returns a path list if goal reached, else None.
        """
        visited = set()
        current = start
        path = [current]
        steps = 0
        while current != goal and steps < max_steps:
            steps += 1
            nbrs = self.grid.neighbors(current)
            # filter out visited nodes (but keep options)
            nbrs = [n for n in nbrs if n not in visited]
            if not nbrs:
                # Allow backtracking if stuck
                nbrs = self.grid.neighbors(current)
                if not nbrs:
                    return None
            # score neighbors by manhattan distance (lower better) + some randomness
            def score(n):
                return abs(n[0] - goal[0]) + abs(n[1] - goal[1]) + self.rng.random() * 2.0
            nbrs.sort(key=score)
            chosen = nbrs[0]
            path.append(chosen)
            visited.add(chosen)
            current = chosen
            if len(path) > self.grid.rows * self.grid.cols * 4:
                return None
        return path if current == goal else None

    def repair_subpath_with_astar(self, a: Position, b: Position) -> Optional[List[Position]]:
        """Return a spatial path between a and b using A* (spatial)."""
        path, cost, nx, tms = a_star_search(self.grid, start=a, goal=b, time_aware=False)
        return path

    # -----------------------
    # Neighbor operators
    # -----------------------
    def neighbor_replace_subpath(self, path: List[Position]) -> Optional[List[Position]]:
        """Pick two indices i < j and try to replace path[i..j] with a fresh A* between endpoints."""
        n = len(path)
        if n < 4:
            return None
        i = self.rng.randrange(0, n - 2)   # up to n-3
        j = self.rng.randrange(i + 2, n)   # at least i+2 so there is room between
        a = path[i]
        b = path[j]
        sub = self.repair_subpath_with_astar(a, b)
        if sub is None:
            return None
        # new_path = path[:i] + sub + path[j+1:]
        new_path = path[:i] + sub + path[j + 1:]
        # optional: remove immediate duplicates
        cleaned = [new_path[0]]
        for p in new_path[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)
        return cleaned

    def neighbor_mutate_waypoint(self, path: List[Position]) -> Optional[List[Position]]:
        """Pick a waypoint k (not start/goal), move it randomly to a neighbor, then repair subpath."""
        n = len(path)
        if n < 5:
            return None
        k = self.rng.randrange(1, n - 1)  # avoid start(0) and goal(n-1)
        prev_pos = path[k - 1]
        next_pos = path[k + 1]
        # pick a random neighbor around path[k]
        cand_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.grid.allow_diagonal:
            cand_dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.rng.shuffle(cand_dirs)
        for dr, dc in cand_dirs:
            new_wp = (path[k][0] + dr, path[k][1] + dc)
            if not self.grid.in_bounds(new_wp):
                continue
            if self.grid.is_blocked_static(new_wp):
                continue
            # attempt to repair prev_pos -> next_pos via A* (this will find a path that goes through new_wp
            # if it's helpful; but we don't force it).
            sub = self.repair_subpath_with_astar(prev_pos, next_pos)
            if sub is None:
                continue
            new_path = path[:k] + sub + path[k + 2:]
            # clean duplicates
            cleaned = [new_path[0]]
            for p in new_path[1:]:
                if p != cleaned[-1]:
                    cleaned.append(p)
            return cleaned
        return None

    def neighbor_shortcut(self, path: List[Position]) -> Optional[List[Position]]:
        """Try to find a shortcut between two nodes that reduces cost."""
        n = len(path)
        if n < 4:
            return None
        # try a few random i,j pairs
        for _ in range(6):
            i = self.rng.randrange(0, n - 2)
            j = self.rng.randrange(i + 2, n)
            a = path[i]
            b = path[j]
            # If a and b are adjacent (manhattan <=1 or diagonal allowed), shortcut directly
            man = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if man == 1 or (self.grid.allow_diagonal and max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1):
                # try direct replacement
                new_path = path[:i + 1] + path[j:]
                cleaned = [new_path[0]]
                for p in new_path[1:]:
                    if p != cleaned[-1]:
                        cleaned.append(p)
                return cleaned
            # else try A* between a and b and see if the returned path is shorter than current subpath
            sub = self.repair_subpath_with_astar(a, b)
            if sub is None:
                continue
            old_sub = path[i:j + 1]
            if self.path_cost(sub) + 0.0 < self.path_cost(old_sub) - 1e-6:
                new_path = path[:i] + sub + path[j + 1:]
                cleaned = [new_path[0]]
                for p in new_path[1:]:
                    if p != cleaned[-1]:
                        cleaned.append(p)
                return cleaned
        return None

    def generate_neighbors(self, path: List[Position], k: int = 8) -> List[List[Position]]:
        """Generate up to k neighbor candidate paths using different operators."""
        neighbors = []
        ops = [self.neighbor_replace_subpath, self.neighbor_mutate_waypoint, self.neighbor_shortcut]
        for _ in range(k):
            op = self.rng.choice(ops)
            try:
                cand = op(path)
            except Exception:
                cand = None
            if cand and len(cand) >= 2:
                neighbors.append(cand)
        # ensure uniqueness (by tupleization)
        unique = []
        seen = set()
        for p in neighbors:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        return unique

    # -----------------------
    # Main planner (hill-climbing + restarts)
    # -----------------------
    def plan(self,
             start: Optional[Position] = None,
             goal: Optional[Position] = None,
             current_time: int = 0,
             max_restarts: int = 5,
             max_iters_per_restart: int = 200,
             patience: int = 40,
             neighbors_per_iter: int = 8,
             collision_penalty: float = 1e6,
             time_limit: Optional[float] = None) -> Tuple[Optional[List[Position]], float, Dict[str, Any]]:
        """
        Run local search. Returns (best_path, best_score, stats)
        stats contains: total_iters, restarts, evaluations, time_s, improvements
        """
        t0 = time.time()
        start_pos = start if start is not None else self.start
        goal_pos = goal if goal is not None else self.goal

        best_path = self.generate_initial_path(start_pos, goal_pos)
        if best_path is None:
            return None, float("inf"), {"reason": "no_initial_path"}

        best_score = self.score(best_path, start_time=current_time, collision_penalty=collision_penalty)
        stats = {
            "total_iters": 0,
            "restarts": 0,
            "evaluations": 0,
            "time_s": 0.0,
            "improvements": 0
        }

        # Outer loop: random restarts
        for restart in range(max_restarts):
            stats["restarts"] += 1
            cur_path = list(best_path)
            cur_score = best_score
            no_improve = 0

            for it in range(max_iters_per_restart):
                # time limit check
                if time_limit is not None and (time.time() - t0) > time_limit:
                    stats["time_s"] = time.time() - t0
                    return best_path, best_score, stats

                stats["total_iters"] += 1
                # generate neighbors
                neighbors = self.generate_neighbors(cur_path, k=neighbors_per_iter)
                stats["evaluations"] += len(neighbors)
                if not neighbors:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                    continue

                # evaluate neighbors, pick best
                best_neighbor = None
                best_neighbor_score = float("inf")
                for nb in neighbors:
                    sc = self.score(nb, start_time=current_time, collision_penalty=collision_penalty)
                    if sc < best_neighbor_score:
                        best_neighbor_score = sc
                        best_neighbor = nb

                # If improvement found, accept (greedy hill climb)
                if best_neighbor_score + 1e-9 < cur_score:
                    cur_path = best_neighbor
                    cur_score = best_neighbor_score
                    no_improve = 0
                    stats["improvements"] += 1
                    # Update global best if better
                    if cur_score + 1e-9 < best_score:
                        best_path = list(cur_path)
                        best_score = cur_score
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

            # End of one restart. If global best found early, we can choose to break early.
            # Perform a random restart (generate a new random initial path) if allowed.
            if restart < max_restarts - 1:
                stats["restarts"] += 0  # counter already incremented above
                # random restart: greedy_random_walk or small perturbation of best
                new_start_path = self.greedy_random_walk(start_pos, goal_pos,
                                                        max_steps=self.grid.rows * self.grid.cols * 4)
                if new_start_path is not None:
                    cur_path = new_start_path
                    cur_score = self.score(cur_path, start_time=current_time, collision_penalty=collision_penalty)
                    # if new random path better than best, update
                    if cur_score < best_score:
                        best_path = list(cur_path)
                        best_score = cur_score
        stats["time_s"] = time.time() - t0
        return best_path, best_score, stats


# -----------------------
# Demo when run directly
# -----------------------
if __name__ == "__main__":
    import sys
    print("LocalSearchPlanner demo on ../maps/map_dynamic.txt")
    grid = Grid.from_files("../maps/map_dynamic.txt", "../maps/dynamic_schedule.json")
    planner = LocalSearchPlanner(grid, seed=1234)
    start_time = 0
    print("Generating initial path via A*...")
    initial = planner.generate_initial_path(grid.start, grid.goal)
    print("Initial path length:", len(initial) if initial else None, "cost:", planner.path_cost(initial) if initial else None)
    print("Running local search (hill-climb+restarts)...")
    best_path, best_score, stats = planner.plan(current_time=start_time,
                                               max_restarts=4,
                                               max_iters_per_restart=150,
                                               patience=30,
                                               neighbors_per_iter=6,
                                               collision_penalty=1e5,
                                               time_limit=10.0)
    print("Best score:", best_score)
    if best_path:
        print("Best path length:", len(best_path))
        print("Best path:", best_path)
    print("Stats:", stats)
