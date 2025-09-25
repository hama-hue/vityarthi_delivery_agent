"""
src/search.py

Provides:
- uniform_cost_search(grid, start=None, goal=None)
- a_star_search(grid, start=None, goal=None, time_aware=False, start_time=0)

Each function returns a tuple:
(path, path_cost, nodes_expanded, planning_time_ms)

- For non-time-aware searches: path is a list of (r,c)
- For time-aware search (time_aware=True): path is a list of ((r,c), t) states.
"""

import heapq
import math
import time
from typing import List, Tuple, Optional, Dict, Any

# Import Grid class (assumes you run this script with cwd = project root and src/ on sys.path)
from grid import Grid

Position = Tuple[int, int]
State = Any  # either Position or (Position, t)


# ---------------------------
# Helpers
# ---------------------------
def reconstruct_path(came_from: Dict[State, Optional[State]], goal_state: State) -> List[State]:
    """Reconstruct path from came_from map. Works for both pos or (pos,t) states."""
    path = []
    cur = goal_state
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def octile_distance(a: Position, b: Position) -> float:
    # octile: good for 8-connected grids (diagonal moves)
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (math.sqrt(2) * min(dx, dy)) + (max(dx, dy) - min(dx, dy))


# ---------------------------
# Uniform-cost search (UCS)
# ---------------------------
def uniform_cost_search(grid: Grid,
                        start: Optional[Position] = None,
                        goal: Optional[Position] = None,
                        max_expansions: Optional[int] = None) -> Tuple[Optional[List[Position]], float, int, float]:
    """
    Standard UCS (Dijkstra) on the spatial grid (ignores time).
    Returns: (path_positions, path_cost, nodes_expanded, planning_time_ms)
    If no path found: path_positions = None, path_cost = float('inf')
    """
    start_time = time.perf_counter()

    if start is None:
        start = grid.start
    if goal is None:
        goal = grid.goal

    # Basic checks
    if grid.is_blocked_static(start) or grid.is_blocked_static(goal):
        return None, float('inf'), 0, 0.0

    frontier = []
    counter = 0
    # heap entry: (g, counter, position)
    heapq.heappush(frontier, (0, counter, start))
    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far: Dict[Position, float] = {start: 0.0}
    nodes_expanded = 0

    while frontier:
        g, _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if max_expansions is not None and nodes_expanded > max_expansions:
            break

        if current == goal:
            planning_time_ms = (time.perf_counter() - start_time) * 1000.0
            path = reconstruct_path(came_from, current)  # list of positions
            return path, cost_so_far[current], nodes_expanded, planning_time_ms

        for neighbor in grid.neighbors(current):
            try:
                step_cost = grid.cell_cost(neighbor)
            except Exception:
                # blocked or invalid
                continue
            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor))
                came_from[neighbor] = current

    # No path found
    planning_time_ms = (time.perf_counter() - start_time) * 1000.0
    return None, float('inf'), nodes_expanded, planning_time_ms


# ---------------------------
# A* search
# ---------------------------
def a_star_search(grid: Grid,
                  start: Optional[Position] = None,
                  goal: Optional[Position] = None,
                  time_aware: bool = False,
                  start_time: int = 0,
                  max_expansions: Optional[int] = None) -> Tuple[Optional[List[Any]], float, int, float]:
    """
    A* search.
    - If time_aware == False: search over positions (pos).
      Returns path list of (r,c).
    - If time_aware == True: search over states (pos, t). Each move increases t by 1.
      Returns path list of ((r,c), t) states.

    Returns: (path, path_cost, nodes_expanded, planning_time_ms)
    """
    t0 = time.perf_counter()

    if start is None:
        start = grid.start
    if goal is None:
        goal = grid.goal

    # Basic checks
    if grid.is_blocked_static(start) or grid.is_blocked_static(goal):
        return None, float('inf'), 0, 0.0

    min_cost = grid.min_cell_cost()
    allow_diag = grid.allow_diagonal

    def heuristic_pos(p: Position) -> float:
        # admissible heuristic based on grid connectivity
        if allow_diag:
            return min_cost * octile_distance(p, goal)
        else:
            return min_cost * manhattan(p, goal)

    nodes_expanded = 0
    counter = 0

    if not time_aware:
        # State = Position
        frontier = []
        g_start = 0.0
        f_start = g_start + heuristic_pos(start)
        # heap: (f, g, counter, pos)
        heapq.heappush(frontier, (f_start, g_start, counter, start))
        came_from: Dict[Position, Optional[Position]] = {start: None}
        cost_so_far: Dict[Position, float] = {start: 0.0}
        while frontier:
            f, g, _, current = heapq.heappop(frontier)
            nodes_expanded += 1
            if max_expansions is not None and nodes_expanded > max_expansions:
                break

            if current == goal:
                planning_time_ms = (time.perf_counter() - t0) * 1000.0
                path = reconstruct_path(came_from, current)
                return path, cost_so_far[current], nodes_expanded, planning_time_ms

            for neighbor in grid.neighbors(current):
                try:
                    step_cost = grid.cell_cost(neighbor)
                except Exception:
                    continue
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    counter += 1
                    f_score = new_cost + heuristic_pos(neighbor)
                    heapq.heappush(frontier, (f_score, new_cost, counter, neighbor))
                    came_from[neighbor] = current

        planning_time_ms = (time.perf_counter() - t0) * 1000.0
        return None, float('inf'), nodes_expanded, planning_time_ms

    else:
        # Time-aware A*: state is (pos, t)
        # We model discrete timesteps. Each move increments t by 1.
        frontier = []
        start_state = (start, int(start_time))
        g_start = 0.0
        f_start = g_start + heuristic_pos(start)
        # heap: (f, g, counter, (pos,t))
        heapq.heappush(frontier, (f_start, g_start, counter, start_state))
        came_from: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {start_state: None}
        cost_so_far: Dict[Tuple[Position, int], float] = {start_state: 0.0}
        visited_best_time: Dict[Tuple[Position, int], float] = {}

        while frontier:
            f, g, _, (current_pos, current_t) = heapq.heappop(frontier)
            nodes_expanded += 1
            if max_expansions is not None and nodes_expanded > max_expansions:
                break

            # If goal reached (pos match). We return the state path including times.
            if current_pos == goal:
                planning_time_ms = (time.perf_counter() - t0) * 1000.0
                path = reconstruct_path(came_from, (current_pos, current_t))
                return path, cost_so_far[(current_pos, current_t)], nodes_expanded, planning_time_ms

            # Expand neighbors: each move to neighbor happens at next time (current_t + 1)
            for neighbor in grid.neighbors(current_pos):
                next_t = current_t + 1
                # check dynamic occupancy: neighbor must be passable at next_t
                if not grid.passable(neighbor, next_t):
                    continue
                try:
                    step_cost = grid.cell_cost(neighbor)
                except Exception:
                    continue
                new_cost = cost_so_far[(current_pos, current_t)] + step_cost
                neighbor_state = (neighbor, next_t)
                if neighbor_state not in cost_so_far or new_cost < cost_so_far[neighbor_state]:
                    cost_so_far[neighbor_state] = new_cost
                    counter += 1
                    f_score = new_cost + heuristic_pos(neighbor)
                    heapq.heappush(frontier, (f_score, new_cost, counter, neighbor_state))
                    came_from[neighbor_state] = (current_pos, current_t)

        planning_time_ms = (time.perf_counter() - t0) * 1000.0
        return None, float('inf'), nodes_expanded, planning_time_ms


# ---------------------------
# Basic demo when run directly
# ---------------------------
if __name__ == "__main__":
    # Demo usage: run UCS and A* on map_small
    print("Loading grid from ../maps/map_small.txt")
    g = Grid.from_files("../maps/map_small.txt", "../maps/dynamic_schedule.json")

    print("\nGrid:")
    g.print_grid()

    print("\nRunning Uniform-Cost Search (UCS)...")
    path_u, cost_u, n_e_u, t_u = uniform_cost_search(g)
    print(f"UCS -> path_cost={cost_u}, nodes_expanded={n_e_u}, time_ms={t_u:.2f}")
    if path_u:
        print("UCS path:", path_u)

    print("\nRunning A* (spatial)...")
    path_a, cost_a, n_e_a, t_a = a_star_search(g)
    print(f"A* -> path_cost={cost_a}, nodes_expanded={n_e_a}, time_ms={t_a:.2f}")
    if path_a:
        print("A* path:", path_a)

    print("\nRunning A* (time-aware) from t=0 ...")
    path_at, cost_at, n_e_at, t_at = a_star_search(g, time_aware=True, start_time=0)
    print(f"A* time-aware -> path_cost={cost_at}, nodes_expanded={n_e_at}, time_ms={t_at:.2f}")
    if path_at:
        print("A* (time-aware) path (pos,t):")
        for st in path_at:
            print(st)

