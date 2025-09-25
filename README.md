# Vityarthi Delivery Agent Project

## 📌 Overview
This project implements a delivery agent navigating a grid world with:
- **Uniform Cost Search (UCS)**
- **A\*** (with heuristics, optional time-aware)
- **Local Search** (hill climbing with random restarts)
- Simulation with moving obstacles
- Visualization (frames & videos)
- Experiment runner and results plotting

## 📂 Project Structure
vityarthi_delivery_agent/
│
├── maps/ # Map files and schedules
│ ├── map_small.txt
│ ├── map_dynamic.txt
│ └── dynamic_schedule.json
│
├── src/
│ ├── init.py
│ ├── map_parser.py
│ ├── grid.py
│ ├── search.py
│ ├── local_search.py
│ ├── agent.py
│ ├── visualize.py
│ ├── cli.py
│ ├── experiments.py
│ └── plot_results.py
│
├── requirements.txt
└── README.md