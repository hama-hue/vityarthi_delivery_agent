# Vityarthi Project – Autonomous Delivery Agent

## Overview
This project implements an autonomous delivery agent navigating a 2D grid city.
The agent delivers packages while handling terrain costs, static obstacles, and
dynamic moving obstacles.

Algorithms implemented:
- Uniform-cost search (UCS)
- A* search with admissible heuristic
- Local search replanning (hill-climbing / simulated annealing)

## Project Structure
- `maps/` → test grid maps
- `src/` → source code
- `tests/` → unit tests
- `reports/` → project report
- `screenshots/` → demo screenshots

## Installation
```bash
pip install -r requirements.txt
