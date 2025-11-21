from typing import List, Tuple, Dict, Optional
import math
import networkx as nx

# Import helpers and params from routing (ajustado para routing.py ao invés de dijkstra.py)
from calculation.routing import (
    haversine,
    nearest_node_to_point,
    _select_best_edge_between,
    geocode_address,
    compress_street_segments,
    build_graph_from_csv,
)

# -------------------------
# Heuristic: admissible lower bound for shortest distance
# -------------------------
def _make_astar_heuristic_shortest(G: nx.DiGraph):
    """
    Build an admissible heuristic for the shortest distance:
    - Simply returns the haversine (straight-line) distance between two nodes.
    - This is always admissible because the actual path distance can never be 
      shorter than the straight-line distance.
    
    This heuristic never overestimates (admissible) and therefore preserves A* optimality.
    """
    def heuristic(u: int, v: int) -> float:
        # Get coordinates
        lat_u = float(G.nodes[u].get("y", 0.0))
        lon_u = float(G.nodes[u].get("x", 0.0))
        lat_v = float(G.nodes[v].get("y", 0.0))
        lon_v = float(G.nodes[v].get("x", 0.0))
        
        # Straight-line distance (in meters) - this is the heuristic
        dist = haversine(lon_u, lat_u, lon_v, lat_v)
        
        return dist
    
    return heuristic

# -------------------------
# Core: A* route by coordinates (shortest distance)
# -------------------------
def route_shortest_a_star_by_coords(
    G: nx.DiGraph,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float
) -> Dict:
    """
    Compute shortest-distance path using A*. Inputs are coordinates (lat, lon).
    Uses 'length' as the edge weight.
    
    Returns a dict containing:
      - start_node, end_node, path_nodes, edges, street_segments (compressed),
        total_length_m, total_time_min, total_fuel_liters
    """
    import time as time_module
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    heuristic = _make_astar_heuristic_shortest(G)
    
    # Mede tempo de execução do algoritmo
    start_time = time_module.perf_counter()
    try:
        path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="length")
    except nx.NetworkXNoPath:
        raise RuntimeError("No path between nodes (A* shortest).")
    execution_time = time_module.perf_counter() - start_time
    
    total_length = total_fuel = total_time_min = 0.0
    edges = []
    street_segments = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = _select_best_edge_between(G, u, v)
        if data is None:
            continue
        
        length = data.get("length", 0.0)
        fuel = data.get("fuel_liters", 0.0)
        time_min = data.get("time_minutes", 0.0)
        name = data.get("name") or "unnamed"
        
        total_length += length
        total_fuel += fuel
        total_time_min += time_min
        edges.append((u, v, data))
        street_segments.append((name, length, fuel, time_min))
    
    compressed = compress_street_segments(street_segments)
    
    return {
        "start_node": start_node,
        "end_node": end_node,
        "path_nodes": path,
        "edges": edges,
        "street_segments": compressed,
        "total_length_m": total_length,
        "total_time_min": total_time_min,
        "total_fuel_liters": total_fuel,
        "execution_time_seconds": execution_time,
    }

# -------------------------
# Convenience: A* route by addresses (geocode + call coords version)
# -------------------------
def route_shortest_a_star_by_addresses(
    G: nx.DiGraph,
    start_addr: str,
    dest_addr: str,
    user_agent: str = "meu_app"
) -> Dict:
    """
    Geocode start/dest addresses and call route_shortest_a_star_by_coords.
    Uses geocode_address imported from routing (consistent with other code).
    """
    start_lat, start_lon, _ = geocode_address(start_addr, user_agent=user_agent)
    dest_lat, dest_lon, _ = geocode_address(dest_addr, user_agent=user_agent)
    
    return route_shortest_a_star_by_coords(G, start_lat, start_lon, dest_lat, dest_lon)

# -------------------------
# Heuristic: admissible lower bound for eco_cost (já mencionado no código do usuário)
# -------------------------
def _make_astar_heuristic_eco(G: nx.DiGraph):
    """
    Build an admissible heuristic for the ecological cost:
    - fuel lower bound: base fuel per meter * straight-line distance (assume no slope and no speed penalty)
    - time penalty lower bound: assume best-case speed (max observed in graph) to get minimum time,
      then convert to liters-equivalent with the same TIME_WEIGHT used in dijkstra preprocessing.
    
    This heuristic never overestimates (admissible) and therefore preserves A* optimality.
    """
    from calculation.routing import BASE_L_PER_100KM, REF_SPEED_KMH, TIME_WEIGHT
    
    base_per_m = BASE_L_PER_100KM / 100000.0  # liters per meter
    
    # compute (conservative) maximum speed observed in the graph
    max_speed_kmh = REF_SPEED_KMH
    for _, _, data in G.edges(data=True):
        try:
            s = float(data.get("maxspeed_kmh", REF_SPEED_KMH))
            if s > max_speed_kmh:
                max_speed_kmh = s
        except Exception:
            continue
    
    max_speed_kmh = max(max_speed_kmh, REF_SPEED_KMH)
    max_speed_m_per_min = max_speed_kmh * 1000.0 / 60.0
    
    # liters-per-minute reference (same base as used in preprocessing)
    ref_speed_m_per_min = REF_SPEED_KMH * 1000.0 / 60.0
    liters_per_min_ref = (BASE_L_PER_100KM / 100000.0) * ref_speed_m_per_min
    
    def heuristic(u: int, v: int) -> float:
        # straight-line distance (in meters)
        lat_u = float(G.nodes[u].get("y", 0.0))
        lon_u = float(G.nodes[u].get("x", 0.0))
        lat_v = float(G.nodes[v].get("y", 0.0))
        lon_v = float(G.nodes[v].get("x", 0.0))
        dist = haversine(lon_u, lat_u, lon_v, lat_v)
        
        # fuel lower bound: assume no slope and no speed penalty
        fuel_lb = base_per_m * dist
        
        # time lower bound: assume travel at max_speed_kmh
        time_min_lb = dist / max_speed_m_per_min if max_speed_m_per_min > 0 else 0.0
        time_penalty_lb = TIME_WEIGHT * time_min_lb * liters_per_min_ref
        
        return fuel_lb + time_penalty_lb
    
    return heuristic

# -------------------------
# Core: A* route by coordinates (ecological)
# -------------------------
def route_ecological_a_star_by_coords(
    G: nx.DiGraph,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float
) -> Dict:
    """
    Compute eco-optimal path using A*. Inputs are coordinates (lat, lon).
    Returns a dict containing:
      - start_node, end_node, path_nodes, edges, street_segments (compressed),
        total_length_m, total_time_min, total_fuel_liters
    """
    import time as time_module
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    heuristic = _make_astar_heuristic_eco(G)
    
    # Mede tempo de execução do algoritmo
    start_time = time_module.perf_counter()
    try:
        path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="eco_cost")
    except nx.NetworkXNoPath:
        raise RuntimeError("No path between nodes (A* eco).")
    execution_time = time_module.perf_counter() - start_time
    
    total_length = total_fuel = total_time_min = 0.0
    edges = []
    street_segments = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = _select_best_edge_between(G, u, v)
        if data is None:
            continue
        
        length = data.get("length", 0.0)
        fuel = data.get("fuel_liters", 0.0)
        time_min = data.get("time_minutes", 0.0)
        name = data.get("name") or "unnamed"
        
        total_length += length
        total_fuel += fuel
        total_time_min += time_min
        edges.append((u, v, data))
        street_segments.append((name, length, fuel, time_min))
    
    compressed = compress_street_segments(street_segments)
    
    return {
        "start_node": start_node,
        "end_node": end_node,
        "path_nodes": path,
        "edges": edges,
        "street_segments": compressed,
        "total_length_m": total_length,
        "total_time_min": total_time_min,
        "total_fuel_liters": total_fuel,
        "execution_time_seconds": execution_time,
    }

# -------------------------
# Convenience: A* route by addresses (ecological)
# -------------------------
def route_ecological_a_star_by_addresses(
    G: nx.DiGraph,
    start_addr: str,
    dest_addr: str,
    user_agent: str = "meu_app"
) -> Dict:
    """
    Geocode start/dest addresses and call route_ecological_a_star_by_coords.
    Uses geocode_address imported from routing (consistent with other code).
    """
    start_lat, start_lon, _ = geocode_address(start_addr, user_agent=user_agent)
    dest_lat, dest_lon, _ = geocode_address(dest_addr, user_agent=user_agent)
    
    return route_ecological_a_star_by_coords(G, start_lat, start_lon, dest_lat, dest_lon)


def calculate_astar_routes(start_addr: str, dest_addr: str):
    G = build_graph_from_csv()
    result_eco = route_ecological_a_star_by_addresses(G, start_addr, dest_addr)
    result_short = route_shortest_a_star_by_addresses(G, start_addr, dest_addr)
    return result_eco, result_short