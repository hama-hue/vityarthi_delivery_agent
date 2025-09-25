import json

def load_grid(file_path):
    """
    Load a grid map from a text file.

    Format:
        First line: rows cols
        Next rows: space-separated values
            - integers >=1 = terrain cost
            - 'S' = start
            - 'G' = goal
            - 'X' = blocked cell
    Returns:
        grid (list of lists, with int or None),
        start (tuple row,col),
        goal (tuple row,col)
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # first line = dimensions
    rows, cols = map(int, lines[0].split())

    grid = []
    start = None
    goal = None

    for r in range(1, rows + 1):
        row_data = lines[r].split()
        row = []
        for c, val in enumerate(row_data):
            if val == "S":
                start = (r - 1, c)
                row.append(1)  # cost = 1
            elif val == "G":
                goal = (r - 1, c)
                row.append(1)
            elif val == "X":
                row.append(None)  # blocked
            else:
                row.append(int(val))
        grid.append(row)

    if start is None or goal is None:
        raise ValueError(f"Map {file_path} missing start (S) or goal (G)")

    return grid, start, goal


def load_schedule(file_path):
    """
    Load dynamic obstacle schedule from JSON.

    JSON format:
    {
      "obstacles": [
        { "id": "car1", "trajectory": [ {"t":0,"pos":[1,3]}, {"t":1,"pos":[1,4]} ] }
      ],
      "time_horizon": 20
    }

    Returns:
        schedule (dict): obstacle_id -> {t: (row,col)}
        time_horizon (int)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    schedule = {}
    for obs in data["obstacles"]:
        obs_id = obs["id"]
        trajectory = {item["t"]: tuple(item["pos"]) for item in obs["trajectory"]}
        schedule[obs_id] = trajectory

    return schedule, data.get("time_horizon", None)


# -------------------
# Quick test when run directly
# -------------------
if __name__ == "__main__":
    grid, start, goal = load_grid("../maps/map_small.txt")
    print("Grid size:", len(grid), "x", len(grid[0]))
    print("Start:", start, "Goal:", goal)

    schedule, horizon = load_schedule("../maps/dynamic_schedule.json")
    print("Schedule loaded:", schedule)
    print("Time horizon:", horizon)
