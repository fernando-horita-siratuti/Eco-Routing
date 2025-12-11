# Computational Complexity Analysis

## Overview

This document presents the time and space complexity analysis of the routing algorithms implemented in this project: **Dijkstra** and **A\***.

## Dijkstra Algorithm

### Time Complexity

**O(E log V)** where:
- **E** = number of edges in the graph
- **V** = number of vertices (nodes) in the graph

#### Justification:

1. **Initialization**: O(V) - initialize distances and predecessors for all vertices
2. **Main loop**: 
   - Each vertex is removed from the priority queue once: O(V log V)
   - Each edge is relaxed once: O(E log V) (due to heap operations)
3. **Path reconstruction**: O(V) in the worst case

**Total complexity**: O(V log V + E log V) = **O(E log V)** (assuming connected graph where E ≥ V-1)

### Space Complexity

**O(V)** where:
- **V** = number of vertices

#### Data structures used:
- `dist`: dictionary with distances of all vertices → O(V)
- `prev`: dictionary with predecessors → O(V)
- `visited`: set of visited vertices → O(V)
- `pq`: priority queue (heap) → O(V) in the worst case

**Total space complexity**: O(V)

### Implementation in the Project

The manual implementation in `calculation/dijkstra.py` uses:
- `heapq` for the priority queue (min-heap)
- Python dictionaries for distances and predecessors
- Set to track visited vertices

## A\* Algorithm

### Time Complexity

**O(E log V)** in the best case with good heuristic, but can be **O(b^d)** in the worst case where:
- **b** = branching factor
- **d** = solution depth

#### Detailed Analysis:

1. **With admissible and consistent heuristic**:
   - A* guarantees finding the optimal path
   - Explores only the nodes necessary to find the solution
   - Complexity: **O(E log V)** (similar to Dijkstra, but explores fewer nodes)

2. **Advantage over Dijkstra**:
   - The heuristic guides the search toward the destination
   - Fewer nodes are explored before finding the solution
   - In practice, A* is faster than Dijkstra for routing problems

3. **In the worst case** (useless heuristic, h(n) = 0):
   - A* degenerates to Dijkstra
   - Complexity: O(E log V)

### Space Complexity

**O(V)** in the best case, **O(b^d)** in the worst case

#### Data structures:
- `open_set`: priority queue with nodes to be explored → O(V) in the best case
- `closed_set`: set of already explored nodes → O(V)
- `g_score`: path cost from start to each node → O(V)
- `f_score`: f(n) = g(n) + h(n) for each node → O(V)

**Total space complexity**: O(V) with good heuristic, O(b^d) in the worst case

### Implemented Heuristics

#### 1. Shortest Distance Heuristic (`_make_astar_heuristic_shortest`)

```python
h(n) = haversine(n, destination)
```

- **Admissibility**: The straight-line distance (haversine) never overestimates the actual path distance
- **Consistency**: The heuristic is consistent (monotonic)
- **Calculation complexity**: O(1) - direct geographic distance calculation

#### 2. Ecological Cost Heuristic (`_make_astar_heuristic_eco`)

```python
h(n) = (base_fuel × straight_distance) + (minimum_time_penalty)
```

Where:
- `base_fuel` = base consumption per meter without slope
- `minimum_time_penalty` = minimum time assuming maximum speed

- **Admissibility**: 
  - Assumes ideal conditions (no slope, maximum speed)
  - Never overestimates the actual cost (which includes slopes and lower speeds)
- **Calculation complexity**: O(1) after preprocessing

## Comparison: Dijkstra vs A\*

### Theoretical Efficiency

| Aspect | Dijkstra | A* |
|---------|----------|-----|
| **Time Complexity** | O(E log V) | O(E log V) with good heuristic |
| **Space Complexity** | O(V) | O(V) with good heuristic |
| **Nodes Explored** | All nodes within distance radius | Only nodes on optimal path + neighborhood |
| **Optimality Guarantee** | Yes | Yes (with admissible heuristic) |

### Practical Efficiency

**A* is more efficient** because:

1. **Fewer nodes explored**: The heuristic guides the search directly to the destination
2. **Directed search**: Instead of exploring in all directions (like Dijkstra), A* focuses on the destination direction
3. **Reduced operations**: Fewer insertions/removals in the priority queue

### Practical Example

For an urban routing graph:
- **Dijkstra**: May explore all nodes within a 5km radius from the origin
- **A***: Explores only nodes along the optimal path + some nearby nodes, resulting in ~30-50% fewer nodes explored

## Ecological Cost Analysis

### Preprocessing Complexity

**O(E)** to calculate ecological costs for all edges:

```python
for each edge (u, v):
    calculate slope → O(1)
    calculate fuel → O(1)
    calculate time → O(1)
    calculate ecological cost → O(1)
```

**Total**: O(E) operations, each with O(1) complexity

### Ecological Cost Formula

```
eco_cost = fuel_liters + time_penalty_equiv_liters

where:
fuel_liters = base_per_m × length × slope_multiplier × speed_factor
time_penalty = TIME_WEIGHT × time_minutes × liters_per_min_ref
```

- **Calculation complexity**: O(1) per edge
- **Preprocessing**: O(E) - done once when building the graph

## Conclusion

Both algorithms have **O(E log V)** time complexity, but **A* is more efficient in practice** due to the heuristic that reduces the number of nodes explored. The implementation uses admissible heuristics that guarantee the optimality of the solutions found.

### References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall.
