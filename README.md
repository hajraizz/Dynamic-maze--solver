# Dynamic maze solver

Interactive visualization and comparison of pathfinding algorithms on dynamic mazes.

## What It Does

Compares 7 search algorithms (BFS, DFS, UCS, DLS, IDS, Greedy, A*) on a dynamically changing maze where:
- Obstacles are added every 10 moves
- Obstacles shift position every 15 moves  
- Start/goal positions can move every 20 moves

Visualizes in real-time: visited nodes, frontier, current path, obstacles. Compares final performance metrics.

## Algorithms Implemented

- **BFS** — Breadth-first search
- **DFS** — Depth-first search with backtracking
- **UCS** — Uniform cost search
- **DLS** — Depth-limited search
- **IDS** — Iterative deepening search
- **Greedy** — Best-first with Manhattan heuristic
- **A*** — A* with Manhattan heuristic

## Key Features

- Live animation during search
- Dynamic maze with changing obstacles/goals
- Performance metrics: execution time, nodes expanded, path length, memory usage
- Comparison charts (bar graphs for all metrics)
- Handles dynamic replanning when obstacles block path

## How to Run

```bash
python mze.py
```

Watch each algorithm run. Close window after each to continue to next algorithm.

At the end: summary table + comparison charts showing which algorithm was most efficient.

## Performance Comparison

Shows metrics for each algorithm:
- Execution time (seconds)
- Nodes expanded (search space explored)
- Final path length
- Peak memory usage

## Technical Details

- 25×25 grid
- Manhattan distance heuristic
- Dynamic obstacle generation maintains path connectivity
- Handles goal shifts by clearing visited set (allows re-exploration)
- Tracks reroutes when path becomes blocked

 
