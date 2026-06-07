import random, time, heapq, tracemalloc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from collections import deque
import sys
sys.setrecursionlimit(10000)

GRID_SIZE  = 25
EMPTY      = 0
OBSTACLE   = 1
ANIM_DELAY = 0.02
STEP_EVERY = 3

C_EMPTY    = 0
C_WALL     = 1
C_VISITED  = 2
C_FRONTIER = 3
C_PATH     = 4
C_START    = 5
C_GOAL     = 6

CMAP = mcolors.ListedColormap([
    'white',
    '#2c2c2c',
    '#b3cde3',
    '#fbb4ae',
    '#fdae61',
    '#1a9641',
    '#d7191c',
])
NORM = mcolors.BoundaryNorm([-0.5 + i for i in range(8)], CMAP.N)


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def in_bounds(r, c, size):
    return 0 <= r < size and 0 <= c < size

def passable(grid, r, c):
    return in_bounds(r, c, len(grid)) and grid[r][c] != OBSTACLE

def get_neighbors(grid, pos):
    r, c = pos
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        if passable(grid, r+dr, c+dc):
            yield (r+dr, c+dc)

def path_exists(grid, start, goal):
    visited = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal: return True
        for nb in get_neighbors(grid, node):
            if nb not in visited:
                visited.add(nb); q.append(nb)
    return False

def generate_maze(size=GRID_SIZE):
    while True:
        grid = [[EMPTY]*size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                if random.random() < 0.30:
                    grid[r][c] = OBSTACLE

        empties = [(r,c) for r in range(size) for c in range(size) if grid[r][c]==EMPTY]
        if len(empties) < 2:
            continue

        random.shuffle(empties)
        start = goal = None
        for i in range(len(empties)):
            for j in range(i+1, len(empties)):
                if manhattan(empties[i], empties[j]) >= 15:
                    start, goal = empties[i], empties[j]
                    break
            if start:
                break

        if not start:
            continue
        if path_exists(grid, start, goal):
            return grid, start, goal


def add_obstacle(grid, start, goal):
    size = len(grid)
    for _ in range(300):
        r, c = random.randint(0,size-1), random.randint(0,size-1)
        if (r,c) not in (start, goal) and grid[r][c] == EMPTY:
            grid[r][c] = OBSTACLE
            if not path_exists(grid, start, goal):
                grid[r][c] = EMPTY
                continue
            return

def move_obstacle(grid, start, goal):
    obs = [(r,c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c]==OBSTACLE]
    random.shuffle(obs)
    for (r,c) in obs:
        adj = [(r+dr,c+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
               if in_bounds(r+dr,c+dc,len(grid)) and grid[r+dr][c+dc]==EMPTY
               and (r+dr,c+dc) not in (start,goal)]
        if adj:
            nr,nc = random.choice(adj)
            grid[r][c]=EMPTY; grid[nr][nc]=OBSTACLE
            if not path_exists(grid, start, goal):
                grid[nr][nc]=EMPTY; grid[r][c]=OBSTACLE
            else:
                return


class DynamicMaze:
    def __init__(self, grid, start, goal, bonus=False):
        self.grid  = [row[:] for row in grid]
        self.start = start
        self.goal  = goal
        self.moves = 0
        self.bonus = bonus

    def tick(self):
        self.moves += 1
        if self.moves % 10 == 0:
            add_obstacle(self.grid, self.start, self.goal)
        if self.moves % 15 == 0:
            move_obstacle(self.grid, self.start, self.goal)
        if self.bonus and self.moves % 20 == 0:
            self._shift_marker()

    def _shift_marker(self):
        attr = random.choice(['start','goal'])
        pos  = self.start if attr=='start' else self.goal
        adj  = [(pos[0]+dr, pos[1]+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                if in_bounds(pos[0]+dr, pos[1]+dc, len(self.grid))
                and self.grid[pos[0]+dr][pos[1]+dc] == EMPTY]
        if adj:
            new = random.choice(adj)
            setattr(self, attr, new)

    def path_blocked(self, path):
        return any(self.grid[r][c] == OBSTACLE for r,c in path)

    def g(self):
        return [row[:] for row in self.grid]


class LiveAnimator:
    def __init__(self, algo_name, size):
        self.name  = algo_name
        self.size  = size
        self.frame = 0

        plt.ion()
        self.fig, (self.ax_maze, self.ax_info) = plt.subplots(
            1, 2, figsize=(14, 7),
            gridspec_kw={'width_ratios': [3, 1]}
        )
        self.fig.suptitle(f"Running: {algo_name}", fontsize=15, fontweight='bold')
        try:
            self.fig.canvas.manager.set_window_title(f"Maze — {algo_name}")
        except Exception:
            pass

        self._display = np.zeros((size, size), dtype=int)
        self._im = self.ax_maze.imshow(self._display, cmap=CMAP, norm=NORM,
                                       interpolation='nearest')
        self.ax_maze.set_xticks(np.arange(-0.5, size, 1), minor=True)
        self.ax_maze.set_yticks(np.arange(-0.5, size, 1), minor=True)
        self.ax_maze.grid(which='minor', color='#cccccc', linewidth=0.3)
        self.ax_maze.tick_params(which='both', bottom=False, left=False,
                                 labelbottom=False, labelleft=False)

        legend_patches = [
            mpatches.Patch(color='white',   label='Empty'),
            mpatches.Patch(color='#2c2c2c', label='Obstacle'),
            mpatches.Patch(color='#b3cde3', label='Visited'),
            mpatches.Patch(color='#fbb4ae', label='Frontier'),
            mpatches.Patch(color='#fdae61', label='Current path'),
            mpatches.Patch(color='#1a9641', label='Start'),
            mpatches.Patch(color='#d7191c', label='Goal'),
        ]
        self.ax_maze.legend(handles=legend_patches, loc='upper right',
                            fontsize=7, framealpha=0.85)

        self.ax_info.axis('off')
        self._info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8)
        )
        plt.tight_layout()
        plt.pause(0.01)

    def _build_display(self, maze, current, visited, frontier, path):
        d = np.zeros((self.size, self.size), dtype=int)
        for r in range(self.size):
            for c in range(self.size):
                if maze.grid[r][c] == OBSTACLE:
                    d[r][c] = C_WALL
        for (r,c) in visited:
            if d[r][c] != C_WALL:
                d[r][c] = C_VISITED
        for (r,c) in frontier:
            if d[r][c] != C_WALL:
                d[r][c] = C_FRONTIER
        for (r,c) in path:
            if d[r][c] != C_WALL:
                d[r][c] = C_PATH
        sr,sc = maze.start
        gr,gc = maze.goal
        d[sr][sc] = C_START
        d[gr][gc] = C_GOAL
        if current and d[current[0]][current[1]] not in (C_START, C_GOAL):
            d[current[0]][current[1]] = C_FRONTIER
        return d

    def update(self, maze, current, visited, frontier, path,
               nodes_exp, reroutes, move_count):
        self.frame += 1
        if self.frame % STEP_EVERY != 0:
            return
        d = self._build_display(maze, current, visited, frontier, path)
        self._im.set_data(d)
        info = (
            f"Algorithm : {self.name}\n"
            f"Move count: {move_count}\n"
            f"Nodes exp : {nodes_exp}\n"
            f"Reroutes  : {reroutes}\n"
            f"Path len  : {len(path)}\n"
            f"Start     : {maze.start}\n"
            f"Goal      : {maze.goal}\n"
            f"Current   : {current}\n"
            f"\nObstacle rules:\n"
            f"  +obstacle every 10 moves\n"
            f"  move obs  every 15 moves\n"
            f"  shift S/G every 20 moves"
        )
        self._info_text.set_text(info)
        self.fig.canvas.draw_idle()
        plt.pause(ANIM_DELAY)

    def finish(self, maze, final_path, result):
        d = self._build_display(maze, None, set(), set(), final_path)
        self._im.set_data(d)
        status = "✓ PATH FOUND" if result.found else "✗ NO PATH"
        self.fig.suptitle(
            f"{self.name}  —  {status}  |  Path={len(final_path)}  "
            f"Nodes={result.nodes_expanded}  Time={result.elapsed:.3f}s",
            fontsize=13, fontweight='bold',
            color='green' if result.found else 'red'
        )
        info = (
            f"FINAL RESULT\n"
            f"           \n"
            f"Algorithm : {self.name}\n"
            f"Found     : {result.found}\n"
            f"Path len  : {len(final_path)}\n"
            f"Move count: {result.move_count}\n"
            f"Nodes exp : {result.nodes_expanded}\n"
            f"Reroutes  : {result.reroutes}\n"
            f"Backtracks: {result.backtrack_moves}\n"
            f"Time      : {result.elapsed:.4f}s\n"
            f"Memory    : {result.peak_mem//1024} KB\n"
            f"\nStart : {maze.start}\n"
            f"Goal  : {maze.goal}"
        )
        self._info_text.set_text(info)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)
        plt.ioff()

    def close(self):
        plt.close(self.fig)


class SearchResult:
    def __init__(self, name):
        self.name            = name
        self.path            = []
        self.found           = False
        self.nodes_expanded  = 0
        self.move_count      = 0
        self.reroutes        = 0
        self.elapsed         = 0.0
        self.peak_mem        = 0
        self.backtrack_moves = 0

#     
# SEARCH ALGORITHMS
#     

#   BFS    

def _bfs_static(grid, start, goal):
    visited = {start}; q = deque([(start, [start])])
    while q:
        node, path = q.popleft()
        if node == goal: return path
        for nb in get_neighbors(grid, node):
            if nb not in visited:
                visited.add(nb); q.append((nb, path+[nb]))
    return None

def bfs(maze: DynamicMaze, anim: LiveAnimator) -> SearchResult:
    res = SearchResult("BFS")
    tracemalloc.start(); t0 = time.perf_counter()

    visited = {maze.start}
    queue   = deque([(maze.start, [maze.start])])

    while queue:
        node, path = queue.popleft()
        res.nodes_expanded += 1
        prev_goal = maze.goal
        maze.tick(); res.move_count += 1

        # FIX: if goal shifted and new goal was already marked visited,
        # un-visit it so BFS can discover it again
        if maze.goal != prev_goal and maze.goal in visited:
            visited.discard(maze.goal)
            queue.append((maze.goal, path + [maze.goal]))

        frontier_set = {item[0] for item in queue}
        anim.update(maze, node, visited, frontier_set, path,
                    res.nodes_expanded, res.reroutes, res.move_count)

        if node == maze.goal:
            if not maze.path_blocked(path):
                res.path = path; res.found = True; break
            else:
                res.reroutes += 1
                sub = _bfs_static(maze.g(), node, maze.goal)
                if sub:
                    res.path = path[:-1] + sub
                    res.found = True
                break

        if maze.path_blocked(path):
            res.reroutes += 1
            sub = _bfs_static(maze.g(), node, maze.goal)
            if sub:
                res.path = path[:-1] + sub
                res.found = True
                break
            # Re-seed from current node; keep visited to avoid re-exploring
            # already-confirmed dead-ends on the *current* grid snapshot
            visited = {node}
            queue = deque([(node, [node])])
            continue

        for nb in get_neighbors(maze.grid, node):
            if nb not in visited:
                visited.add(nb); queue.append((nb, path+[nb]))

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   DFS    

def _dfs_rec(maze, node, goal, path_set, path, res, anim, bt=0):
    path = path + [node]
    res.nodes_expanded += 1
    maze.tick(); res.move_count += 1

    anim.update(maze, node, path_set, set(), path,
                res.nodes_expanded, res.reroutes, res.move_count)

    if node == maze.goal:
        if not maze.path_blocked(path):
            res.path = path; res.found = True; return True, path, bt
        else:
            res.reroutes += 1
            sub = _bfs_static(maze.g(), node, maze.goal)
            if sub:
                res.path = path[:-1] + sub
                res.found = True
            return True, path, bt

    for nb in get_neighbors(maze.grid, node):
        if nb not in path_set:
            path_set.add(nb)
            found, p, bt = _dfs_rec(maze, nb, maze.goal, path_set, path, res, anim, bt)
            if found:
                return True, p, bt
            path_set.discard(nb)
            bt += 1
    return False, path, bt

def dfs(maze: DynamicMaze, anim: LiveAnimator) -> SearchResult:
    res = SearchResult("DFS")
    tracemalloc.start(); t0 = time.perf_counter()

    path_set = {maze.start}
    found, path, bt = _dfs_rec(maze, maze.start, maze.goal, path_set, [], res, anim)
    res.path = path if found else []
    res.found = found; res.backtrack_moves = bt

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   UCS    

def ucs(maze: DynamicMaze, anim: LiveAnimator) -> SearchResult:
    res = SearchResult("UCS")
    tracemalloc.start(); t0 = time.perf_counter()

    ctr = 0
    pq  = [(0, ctr, maze.start, [maze.start])]
    best = {}

    while pq:
        cost, _, node, path = heapq.heappop(pq)

        if node in best and best[node] <= cost:
            continue
        best[node] = cost

        res.nodes_expanded += 1
        maze.tick(); res.move_count += 1

        frontier_set = {item[2] for item in pq}
        anim.update(maze, node, set(best.keys()), frontier_set, path,
                    res.nodes_expanded, res.reroutes, res.move_count)

        if node == maze.goal:
            if not maze.path_blocked(path):
                res.path = path; res.found = True; break
            else:
                res.reroutes += 1
                sub = _bfs_static(maze.g(), node, maze.goal)
                if sub:
                    res.path = path[:-1] + sub
                    res.found = True
                break

        for nb in get_neighbors(maze.grid, node):
            nc = cost + 1
            if nb not in best or best[nb] > nc:
                ctr += 1; heapq.heappush(pq, (nc, ctr, nb, path+[nb]))

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   DLS    

def _dls_rec(maze, node, goal, limit, path_set, path, res, anim, bt=0):
    path = path + [node]
    res.nodes_expanded += 1
    maze.tick(); res.move_count += 1

    anim.update(maze, node, path_set, set(), path,
                res.nodes_expanded, res.reroutes, res.move_count)

    if node == maze.goal:
        if not maze.path_blocked(path):
            res.path = path; res.found = True; return True, path, bt
        else:
            res.reroutes += 1
            sub = _bfs_static(maze.g(), node, maze.goal)
            if sub:
                res.path = path[:-1] + sub
                res.found = True
            return True, path, bt
    if limit == 0:
        return False, path, bt

    for nb in get_neighbors(maze.grid, node):
        if nb not in path_set:
            path_set.add(nb)
            found, p, bt = _dls_rec(maze, nb, maze.goal, limit-1, path_set,
                                    path, res, anim, bt)
            if found:
                return True, p, bt
            path_set.discard(nb)
            bt += 1
    return False, path, bt

def dls(maze: DynamicMaze, anim: LiveAnimator, depth_limit=40) -> SearchResult:
    res = SearchResult(f"DLS(limit={depth_limit})")
    tracemalloc.start(); t0 = time.perf_counter()

    path_set = {maze.start}
    found, path, bt = _dls_rec(maze, maze.start, maze.goal, depth_limit - 1,
                                path_set, [], res, anim)
    res.path = path if found else []
    res.found = found; res.backtrack_moves = bt

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   IDS    

def ids(maze: DynamicMaze, anim: LiveAnimator, max_depth=80) -> SearchResult:
    res = SearchResult("IDS")
    tracemalloc.start(); t0 = time.perf_counter()

    for depth in range(max_depth):
        path_set = {maze.start}
        found, path, bt = _dls_rec(maze, maze.start, maze.goal, depth,
                                   path_set, [], res, anim)
        res.backtrack_moves += bt
        if found:
            res.path = path; res.found = True; break

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   Greedy  

def greedy(maze: DynamicMaze, anim: LiveAnimator) -> SearchResult:
    res = SearchResult("Greedy")
    tracemalloc.start(); t0 = time.perf_counter()

    ctr = 0
    pq  = [(manhattan(maze.start, maze.goal), ctr, maze.start, [maze.start])]
    visited = set()

    while pq:
        h, _, node, path = heapq.heappop(pq)
        if node in visited: continue
        visited.add(node)
        res.nodes_expanded += 1

        prev_goal = maze.goal
        maze.tick(); res.move_count += 1

        # FIX: if goal shifted, recompute heuristics by clearing visited so
        # we can re-push neighbors with updated distances toward the new goal
        if maze.goal != prev_goal:
            visited.discard(maze.goal)

        frontier_set = {item[2] for item in pq}
        anim.update(maze, node, visited, frontier_set, path,
                    res.nodes_expanded, res.reroutes, res.move_count)

        if node == maze.goal:
            if not maze.path_blocked(path):
                res.path = path; res.found = True; break
            else:
                res.reroutes += 1
                sub = _bfs_static(maze.g(), node, maze.goal)
                if sub:
                    res.path = path[:-1] + sub
                    res.found = True
                break

        if maze.path_blocked(path):
            res.reroutes += 1
            # Restart greedy search from current node with fresh visited set
            visited = set(path)

        for nb in get_neighbors(maze.grid, node):
            if nb not in visited:
                ctr += 1
                # FIX: always compute heuristic against current (possibly
                # shifted) goal so new entries in the heap are correct
                heapq.heappush(pq, (manhattan(nb, maze.goal), ctr, nb, path+[nb]))

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res


#   A*  

def _reconstruct(came_from, node):
    """Walk came_from back to the root (None sentinel) and return the path."""
    path = []
    while node is not None:
        path.append(node)
        node = came_from.get(node)  # None when we reach the start sentinel
    return path[::-1]

def astar(maze: DynamicMaze, anim: LiveAnimator) -> SearchResult:
    res = SearchResult("A*")
    tracemalloc.start(); t0 = time.perf_counter()

    ctr       = 0
    came_from = {maze.start: None}
    best      = {}
    pq        = [(manhattan(maze.start, maze.goal), 0, ctr, maze.start)]

    while pq:
        f, g, _, node = heapq.heappop(pq)

        if node in best and best[node] <= g:
            continue
        best[node] = g

        res.nodes_expanded += 1
        prev_goal = maze.goal
        maze.tick(); res.move_count += 1

        if maze.goal != prev_goal:
            best.clear()
            came_from = {maze.start: None}
            ctr += 1
            heapq.heappush(pq, (manhattan(maze.start, maze.goal), 0, ctr, maze.start))
            continue  # don't process this node further — goal has moved

        current_path = _reconstruct(came_from, node)
        frontier_set = {item[3] for item in pq}
        anim.update(maze, node, set(best.keys()), frontier_set, current_path,
                    res.nodes_expanded, res.reroutes, res.move_count)

        if node == maze.goal:
            if not maze.path_blocked(current_path):
                res.path = current_path; res.found = True; break
            else:
                res.reroutes += 1
                sub = _bfs_static(maze.g(), node, maze.goal)
                if sub:
                    res.path = current_path[:-1] + sub
                    res.found = True
                break

        if maze.path_blocked(current_path):
            res.reroutes += 1
            sub = _bfs_static(maze.g(), node, maze.goal)
            if sub:
                res.path = current_path[:-1] + sub
                res.found = True
                break
            # Full restart
            came_from = {maze.start: None}
            best.clear()
            ctr += 1
            heapq.heappush(pq, (manhattan(maze.start, maze.goal), 0, ctr, maze.start))
            continue

        for nb in get_neighbors(maze.grid, node):
            ng = g + 1
            if nb not in best or best[nb] > ng:
                ctr += 1
                came_from[nb] = node
                heapq.heappush(pq, (ng + manhattan(nb, maze.goal), ng, ctr, nb))

    res.elapsed = time.perf_counter() - t0
    _, res.peak_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return res
#     
# RESULTS DISPLAY
#     

def show_comparison(results):
    names    = [r.name for r in results]
    times    = [r.elapsed for r in results]
    nodes    = [r.nodes_expanded for r in results]
    path_len = [len(r.path) for r in results]
    mem      = [r.peak_mem / 1024 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Algorithm Performance Comparison", fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    def bar(ax, vals, ylabel, title):
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                    f'{v:.3g}', ha='center', va='bottom', fontsize=8)

    bar(axes[0][0], times,    "Seconds",  "Execution Time")
    bar(axes[0][1], nodes,    "Count",    "Nodes Expanded")
    bar(axes[1][0], path_len, "Steps",    "Final Path Length")
    bar(axes[1][1], mem,      "KB",       "Peak Memory Usage")

    plt.tight_layout()
    plt.show()


def print_result(r: SearchResult):
    print(f"  {' '*56}")
    print(f"  {r.name}")
    print(f"    Found       : {r.found}")
    print(f"    Path length : {len(r.path)}")
    print(f"    Move count  : {r.move_count}")
    print(f"    Nodes exp.  : {r.nodes_expanded}")
    print(f"    Reroutes    : {r.reroutes}")
    print(f"    Backtracks  : {r.backtrack_moves}")
    print(f"    Time        : {r.elapsed:.5f} s")
    print(f"    Memory      : {r.peak_mem//1024} KB")
    if r.path:
        sample = r.path[:4] + (['...'] if len(r.path)>6 else []) + r.path[-2:]
        print(f"    Path sample : {' → '.join(str(p) for p in sample)}")


def main():
    print("=" * 60)
    print("  DYNAMIC MAZE — SEARCH ALGORITHM COMPARISON")
    print("=" * 60)

    random.seed(42)
    grid, start, goal = generate_maze(GRID_SIZE)

    print(f"\nMaze {GRID_SIZE}×{GRID_SIZE}  Start={start}  Goal={goal}  "
          f"Distance={manhattan(start,goal)}\n")

    algorithms = [
        ("A*",              astar,  {}),
        ("IDS",             ids,    {"max_depth": 60}),
        ("BFS",             bfs,    {}),
        ("DFS",             dfs,    {}),
        ("UCS",             ucs,    {}),
        ("DLS (limit=50)",  dls,    {"depth_limit": 50}),
        ("Greedy",          greedy, {}),
    ]

    results = []
    for name, fn, kwargs in algorithms:
        print(f"▶  Running {name} ...")
        maze = DynamicMaze([row[:] for row in grid], start, goal, bonus=True)
        anim = LiveAnimator(name, GRID_SIZE)
        res  = fn(maze, anim, **kwargs)
        anim.finish(maze, res.path, res)
        print_result(res)
        results.append(res)
        print(f"   [Close the '{name}' window to continue]\n")
        plt.show(block=True)
        anim.close()

    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    hdr = f"{'Algorithm':<22}{'Found':<8}{'Path':<8}{'Nodes':<10}{'Time(s)':<12}{'Mem(KB)'}"
    print(hdr); print(" "*len(hdr))
    for r in results:
        print(f"{r.name:<22}{str(r.found):<8}{len(r.path):<8}"
              f"{r.nodes_expanded:<10}{r.elapsed:<12.4f}{r.peak_mem//1024}")

    show_comparison(results)


if __name__ == "__main__":
    main()
