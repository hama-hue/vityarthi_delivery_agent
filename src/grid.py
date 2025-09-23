"""
src/grid.py

Grid environment for the Vityarthi delivery agent project.

Provides:
- Grid class that stores terrain costs, start/goal, and dynamic obstacle schedule
- Helpers: in_bounds, passable (time-aware), neighbors (4- or 8-connected),
  cell_cost, min_cell_cost, pretty printing
- Classmethod to load from map + schedule files using the provided map_parser.py
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

# Import the parser in the same directory (src/)
# When you run "python src/grid.py" this works because src/ is sys.path[0]
from map_parser import load_grid, load_schedule

Position = Tuple[int, int]  # (row, col)


class Grid:
    def __init__(
        self,
        grid: List[List[Optional[int]]],
        start: Position,
        goal: Position,
        schedule: Optional[Dict[str, Dict[int, Position]]] = None,
        time_horizon: Optional[int] = None,
        allow_diagonal: bool = False,
    ):
        """
        grid: 2D list, each cell is either int (cost >= 1) or None (blocked)
        start, goal: (row, col)
        schedule: dict mapping obstacle_id -> {t: (row, col), ...}
        time_horizon: optional int (max time to consider)
        allow_diagonal: if True, neighbors includes 8 directions
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.allow_diagonal = allow_diagonal

        # store schedule and time horizon
        self.schedule = schedule or {}
        self.time_horizon = time_horizon

        # Precompute occupancy: map time -> set of positions occupied at that time
        # occupancy[t] = {(r,c), ...}
        self.occupancy: Dict[int, Set[Position]] = defaultdict(set)
        for obs_id, traj in self.schedule.items():
            for t, pos in traj.items():
                self.occupancy[int(t)].add(tuple(pos))

    # --------------------
    # Basic grid queries
    # --------------------
    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked_static(self, pos: Position) -> bool:
        """True if cell is a static obstacle or out of bounds."""
        if not self.in_bounds(pos):
            return True
        return self.grid[pos[0]][pos[1]] is None

    def is_occupied_by_obstacle(self, pos: Position, t: Optional[int]) -> bool:
        """Return True if a moving obstacle occupies pos at time t."""
        if t is None:
            return False
        return tuple(pos) in self.occupancy.get(int(t), set())

    def passable(self, pos: Position, t: Optional[int] = None) -> bool:
        """
        Is the cell passable at time t?
        - False if statically blocked (X)
        - False if an obstacle occupies the cell at that time
        - Otherwise True
        """
        if self.is_blocked_static(pos):
            return False
        if t is not None and self.is_occupied_by_obstacle(pos, t):
            return False
        return True

    def cell_cost(self, pos: Position) -> int:
        """Return movement cost for entering pos. Raises ValueError if blocked or OOB."""
        if not self.in_bounds(pos):
            raise IndexError(f"Position out of bounds: {pos}")
        val = self.grid[pos[0]][pos[1]]
        if val is None:
            raise ValueError(f"Cell {pos} is blocked (None).")
        # Ensure it's an int >= 1
        return int(val)

    def min_cell_cost(self) -> int:
        """Return minimum positive cell cost across the grid (useful for heuristics)."""
        min_cost = None
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.grid[r][c]
                if v is None:
                    continue
                if min_cost is None or int(v) < min_cost:
                    min_cost = int(v)
        return min_cost if min_cost is not None else 1

    # --------------------
    # Neighbours
    # --------------------
    def neighbors(self, pos: Position) -> List[Position]:
        """
        Return list of neighbor positions (4-connected by default).
        Note: this method does NOT check dynamic occupancy (time). Use passable(pos, t)
        when planning in time-expanded space.
        """
        r, c = pos
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        result = []
        for dr, dc in directions:
            np = (r + dr, c + dc)
            if not self.in_bounds(np):
                continue
            if self.grid[np[0]][np[1]] is None:
                # statically blocked
                continue
            result.append(np)
        return result

    # --------------------
    # Utilities
    # --------------------
    def print_grid(self):
        """Pretty-print the grid: numbers for cost, X for blocked, S/G markers."""
        symbol_map = {}
        symbol_map[self.start] = "S"
        symbol_map[self.goal] = "G"
        for r in range(self.rows):
            parts = []
            for c in range(self.cols):
                pos = (r, c)
                if pos in symbol_map:
                    parts.append(symbol_map[pos])
                else:
                    v = self.grid[r][c]
                    parts.append("X" if v is None else str(v))
            print(" ".join(parts))

    @classmethod
    def from_files(cls, map_file: str, schedule_file: Optional[str] = None, allow_diagonal: bool = False):
        """
        Convenience loader: load grid and schedule from files (paths relative to project root).
        Example:
            Grid.from_files("maps/map_small.txt", "maps/dynamic_schedule.json")
        """
        grid, start, goal = load_grid(map_file)
        schedule = {}
        horizon = None
        if schedule_file:
            schedule, horizon = load_schedule(schedule_file)
        return cls(grid=grid, start=start, goal=goal, schedule=schedule, time_horizon=horizon, allow_diagonal=allow_diagonal)


# -------------------
# Quick demo / smoke test when run directly
# -------------------
if __name__ == "__main__":
    # When you run: python src/grid.py  (from project root)
    # the relative paths below assume repo structure:
    # repo/
    #   maps/
    #   src/
    print("Loading map ../maps/map_small.txt and schedule ../maps/dynamic_schedule.json (if present)...")
    try:
        grid = Grid.from_files("../maps/map_small.txt", "../maps/dynamic_schedule.json")
    except Exception as e:
        print("Error loading map/schedule:", e)
        raise

    print("\nGrid size:", grid.rows, "x", grid.cols)
    print("Start:", grid.start, "Goal:", grid.goal)
    print("\nPretty printed grid:")
    grid.print_grid()

    print("\nNeighbors of start:", grid.neighbors(grid.start))
    print("Min cell cost (for heuristics):", grid.min_cell_cost())

    # Example time-aware checks
    sample_pos = (1, 1)
    print(f"\nIs sample_pos {sample_pos} statically blocked? ->", grid.is_blocked_static(sample_pos))
    for t_check in range(0, 6):
        print(f"At t={t_check} passable({sample_pos})? ->", grid.passable(sample_pos, t_check))

