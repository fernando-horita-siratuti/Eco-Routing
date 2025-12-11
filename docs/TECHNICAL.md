# Detailed Technical Documentation

## Table of Contents

1. [A* Algorithm Heuristics](#a-algorithm-heuristics)
2. [Heuristic Admissibility](#heuristic-admissibility)
3. [Ecological Cost Formula](#ecological-cost-formula)
4. [Slope Calculation](#slope-calculation)
5. [Haversine Formula](#haversine-formula)
6. [Theoretical Comparison: Dijkstra vs A*](#theoretical-comparison-dijkstra-vs-a)

---

## A* Algorithm Heuristics

### Heuristic Definition

A **heuristic** h(n) is a function that estimates the cost of the cheapest path from node `n` to the goal node. In the context of A*, the heuristic is used to guide the search more efficiently.

### Shortest Distance Heuristic

**Function**: `_make_astar_heuristic_shortest(G)`

```python
h(n) = haversine(current_node, destination_node)
```

#### Explanation:

- Uses the **straight-line distance** (haversine) between the current node and the destination
- This is always an **underestimate** of the actual path distance
- The actual path distance can never be less than the straight-line distance

#### Why it is admissible:

The straight-line distance is the shortest possible path between two points. Any real path in the street graph will be greater than or equal to this distance, ensuring that the heuristic never overestimates the actual cost.

### Ecological Cost Heuristic

**Function**: `_make_astar_heuristic_eco(G)`

```python
h(n) = fuel_lower_bound + time_penalty_lower_bound

where:
fuel_lower_bound = base_per_m × straight_distance
time_penalty_lower_bound = TIME_WEIGHT × minimum_time × liters_per_min_ref
minimum_time = straight_distance / maximum_speed
```

#### Detailed Explanation:

1. **Fuel Lower Bound**:
   - Assumes base consumption without slope (ideal condition)
   - Multiplies by straight-line distance
   - Never overestimates because it ignores slopes that increase consumption

2. **Time Penalty Lower Bound**:
   - Assumes maximum speed observed in the graph
   - Calculates minimum possible time to traverse the distance
   - Converts to liters equivalent using `TIME_WEIGHT`

3. **Sum of Lower Bounds**:
   - The sum of two lower bounds is always a lower bound of the total cost
   - Guarantees that the heuristic never overestimates the actual ecological cost

---

## Heuristic Admissibility

### Admissible Heuristic Definition

A heuristic **h(n)** is **admissible** if it never overestimates the actual cost of the optimal path from node `n` to the goal. That is:

```
h(n) ≤ h*(n)
```

where `h*(n)` is the actual cost of the optimal path.

### Admissibility Proof

#### Distance Heuristic:

**Proposition**: `haversine(n, destination) ≤ actual_path_distance(n, destination)`

**Proof**:
- The straight-line distance is the shortest possible path between two points in Euclidean space
- Any path in the street graph follows streets, which are curves and not straight lines
- Therefore, the actual distance is always ≥ straight-line distance
- **Conclusion**: The heuristic is admissible ✓

#### Ecological Cost Heuristic:

**Proposition**: `h_eco(n) ≤ actual_ecological_cost(n, destination)`

**Proof**:

1. **Fuel**: 
   - Heuristic assumes: `base_per_m × distance` (without slope)
   - Reality: `base_per_m × distance × slope_multiplier × speed_factor`
   - Since `slope_multiplier ≥ 1` and `speed_factor ≥ 1`, the actual cost is always ≥ heuristic

2. **Time**:
   - Heuristic assumes: maximum speed in the graph
   - Reality: speed may be lower
   - Actual time ≥ minimum assumed time

3. **Sum**:
   - Since both components are lower bounds, the sum is also a lower bound
   - **Conclusion**: The heuristic is admissible ✓

### Importance of Admissibility

An **admissible** heuristic guarantees that A* finds the **optimal path**. If the heuristic overestimates, the algorithm may discard the optimal path prematurely.

---

## Ecological Cost Formula

### Complete Formula

```
eco_cost = fuel_liters + time_penalty_equiv_liters
```

### Component 1: Fuel (fuel_liters)

```
fuel_liters = base_per_m × length × slope_multiplier × speed_factor
```

#### Parameters:

- **base_per_m**: Base fuel consumption per meter (L/m)
  - Calculated as: `BASE_L_PER_100KM / 100000.0`
  - Example: 10 L/100km = 0.0001 L/m

- **length**: Edge length in meters

- **slope_multiplier**: Multiplier due to slope
  ```
  slope_multiplier = 1.0 + (SLOPE_COEF × uphill)
  ```
  where:
  - `uphill = max(slope, 0.0)` (only uphill increases consumption)
  - `SLOPE_COEF = 10.0` (slope sensitivity coefficient)
  
  **Interpretation**: A slope of 0.1 (10%) increases consumption by 100%

- **speed_factor**: Factor due to speed
  ```
  speed_factor = 1.0 + SPEED_PENALTY_COEF × ((speed_kmh - REF_SPEED_KMH) / REF_SPEED_KMH)²
  ```
  where:
  - `SPEED_PENALTY_COEF = 0.2`
  - `REF_SPEED_KMH = 50.0` km/h (reference speed)
  
  **Interpretation**: Speeds very different from reference increase consumption (quadratic penalty)

### Component 2: Time Penalty (time_penalty_equiv_liters)

```
time_penalty_equiv_liters = TIME_WEIGHT × time_minutes × liters_per_min_ref
```

#### Parameters:

- **TIME_WEIGHT**: Weight of time relative to fuel
  - Value: `0.5`
  - Meaning: 1 extra minute = 0.5 equivalent liters

- **time_minutes**: Travel time in minutes
  ```
  time_minutes = length / (speed_kmh × 1000.0 / 60.0)
  ```

- **liters_per_min_ref**: Consumption per minute at reference speed
  ```
  liters_per_min_ref = base_per_m × ref_speed_m_per_min
  ```

### Formula Justification

1. **Real Fuel**: Consumption increases with slope (basic physics) and varies with speed (engine efficiency)

2. **Time-Fuel Equivalence**: Time also has value (opportunity, wear, etc.), so it is converted to a common unit (equivalent liters)

3. **Multi-objective Optimization**: The formula allows balancing fuel economy vs. travel time through the `TIME_WEIGHT` parameter

---

## Slope Calculation

### Formula Used

Slope is calculated using the `street_steepness()` function which uses:

```python
grade = dh / dist_horizontal
inclination_deg = atan2(dh, dist_horizontal) × (180/π)
```

### Components:

1. **dh**: Elevation difference (meters)
   ```
   dh = h2 - h1
   ```

2. **dist_horizontal**: Horizontal distance (meters)
   - Calculated using `horizontal_displacement_m()` which considers Earth's curvature

3. **grade**: Slope ratio (dimensionless)
   - `grade = 0.1` means 10% slope
   - `grade = 0.05` means 5% slope

4. **inclination_deg**: Slope angle in degrees
   ```
   inclination_deg = atan2(dh, dist_horizontal) × (180/π)
   ```

### Mathematical Justification

#### Why use `atan2`?

- `atan2(y, x)` returns the correct angle in all quadrants
- Considers the sign of `dh` (uphill vs. downhill)
- More robust than `atan(dh/dist_h)` when `dist_h` may be zero

#### Why use horizontal distance?

- Horizontal distance is the actual distance traveled on the surface
- Slope is defined as the ratio between vertical elevation and horizontal displacement
- This is more accurate than using 3D distance to calculate slope

### Impact on Ecological Cost

Slope affects consumption through `slope_multiplier`:

```
slope_multiplier = 1.0 + (SLOPE_COEF × uphill)
```

- **Uphill** (`uphill > 0`): Increases consumption exponentially
- **Downhill** (`uphill ≤ 0`): Does not increase consumption (only uphill is penalized)

**Example**:
- Slope of 5% (`grade = 0.05`): `slope_multiplier = 1.0 + (10.0 × 0.05) = 1.5`
- This means **50% more consumption** on this edge

---

## Haversine Formula

### Complete Formula

The haversine distance between two points on Earth's surface is calculated as:

```python
R = 6371000.0  # Earth's radius in meters
φ1, φ2 = lat1, lat2 in radians
λ1, λ2 = lon1, lon2 in radians

a = sin²((φ2 - φ1)/2) + cos(φ1) × cos(φ2) × sin²((λ2 - λ1)/2)
d = 2 × R × atan2(√a, √(1-a))
```

### Step-by-Step Explanation

1. **Conversion to radians**: Latitudes and longitudes are converted from degrees to radians

2. **Differences**: Calculates latitude (`dφ`) and longitude (`dλ`) differences

3. **Haversine formula**: Calculates `a`, which is an intermediate measure based on the haversine formula

4. **Angular distance**: `atan2(√a, √(1-a))` returns the angular distance between the points

5. **Linear distance**: Multiplies the angular distance by Earth's radius to get distance in meters

### Why Haversine?

- **Accuracy**: Considers Earth's curvature, essential for geographic distances
- **Suitable for routing**: For urban distances (< 100km), accuracy is excellent
- **Efficiency**: O(1) calculation, very fast

### Limitations

- Assumes Earth as a perfect sphere (actually an ellipsoid)
- For very long distances (> 1000km), consider using more precise formulas (Vincenty)
- For our case (urban routing), accuracy is more than sufficient

---

## Theoretical Comparison: Dijkstra vs A*

### Algorithmic Structure

#### Dijkstra:

```
1. Initialize distances: dist[start] = 0, others = ∞
2. Create priority queue with all nodes
3. While queue not empty:
   a. Remove node u with smallest distance
   b. For each neighbor v of u:
      - If dist[u] + weight(u,v) < dist[v]:
        - Update dist[v]
        - Update queue
4. Return reconstructed path
```

#### A*:

```
1. Initialize: g[start] = 0, f[start] = h(start)
2. Create open_set with start, empty closed_set
3. While open_set not empty:
   a. Remove node u with smallest f(u) = g(u) + h(u)
   b. If u == destination: return path
   c. Add u to closed_set
   d. For each neighbor v of u:
      - If v in closed_set: skip
      - Calculate g_tentative = g[u] + weight(u,v)
      - If g_tentative < g[v] or v not in open_set:
        - Update g[v] and f[v] = g[v] + h(v)
        - Add/update v in open_set
4. Return reconstructed path
```

### Main Differences

| Aspect | Dijkstra | A* |
|---------|----------|-----|
| **Selection criterion** | Smallest distance from start | Smallest f(n) = g(n) + h(n) |
| **Search direction** | Circular expansion | Directed toward destination |
| **Nodes explored** | All within radius | Only optimal path + neighborhood |
| **Heuristic** | Does not use | Uses h(n) to guide search |

### Theoretical Example

Consider a routing graph where:
- Origin: point A
- Destination: point B (10km north)
- Graph: street grid

**Dijkstra explores**:
- All nodes within a 10km radius circle centered at A
- May explore ~100-200 nodes before finding B

**A* explores**:
- Nodes along the optimal path (north direction)
- Nodes near the optimal path
- May explore ~30-50 nodes before finding B

**Result**: A* is ~2-3x faster in practice

### Theoretical Guarantees

#### Dijkstra:
- **Guarantee**: Always finds the shortest path
- **Condition**: Non-negative weights

#### A*:
- **Guarantee**: Always finds the optimal path
- **Condition**: Admissible heuristic (never overestimates)
- **Optimization**: With consistent heuristic, each node is explored only once

### When to Use Each Algorithm

**Use Dijkstra when**:
- There is no information about the destination
- Need to calculate paths to all destinations
- Heuristic is not available or not reliable

**Use A* when**:
- There is information about the destination (coordinates)
- Need better performance
- Can construct an admissible heuristic
- **Our case**: Routing with known coordinates → A* is ideal

---

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. Chapter 24: Single-Source Shortest Paths.

- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Prentice Hall. Chapter 3: Solving Problems by Searching.

- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
