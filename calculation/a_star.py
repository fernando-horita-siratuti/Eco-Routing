from typing import Dict, Tuple, List, Callable
import logging
import math
import networkx as nx

from calculation.dijkstra import (
    haversine,
    nearest_node_to_point,
    _select_best_edge_between,
    geocode_address,
    compress_street_segments,
    build_graph_from_csv,
)

logger = logging.getLogger(__name__)

def _make_astar_heuristic_shortest(G: nx.DiGraph) -> Callable[[int, int], float]:
    """
    Build an admissible heuristic for the shortest distance:
    - Simply returns the haversine (straight-line) distance between two nodes.
    - This is always admissible because the actual path distance can never be 
      shorter than the straight-line distance.
    
    This heuristic never overestimates (admissible) and therefore preserves A* optimality.
    """
    def heuristic(u: int, v: int) -> float:
        lat_u = float(G.nodes[u].get("y", 0.0))
        lon_u = float(G.nodes[u].get("x", 0.0))
        lat_v = float(G.nodes[v].get("y", 0.0))
        lon_v = float(G.nodes[v].get("x", 0.0))
        
        dist = haversine(lon_u, lat_u, lon_v, lat_v)
        
        return dist
    
    return heuristic


def astar_manual_with_metrics(
    G: nx.DiGraph,
    start: int,
    target: int,
    heuristic_func: Callable[[int, int], float],
    weight: str = 'length'
) -> Tuple[List[int], float, int]:
    """
    Manual implementation of A* algorithm with performance metrics.
    Returns (path, total_cost, nodes_visited_count).
    
    Args:
        G: Directed graph
        start: Origin node
        target: Destination node
        heuristic_func: Heuristic function h(n, target)
        weight: Edge attribute to use as weight ('length', 'eco_cost', etc.)
    
    Returns:
        Tuple of (path, total_cost, nodes_visited_count)
    """
    import heapq
    
    g_score = {node: float('inf') for node in G.nodes()}
    g_score[start] = 0.0
    
    f_score = {node: float('inf') for node in G.nodes()}
    f_score[start] = heuristic_func(start, target)

    came_from = {node: None for node in G.nodes()}
    
    open_set = [(f_score[start], start)]
    
    closed_set = set()
    
    nodes_visited_count = 0
    
    while open_set:
        current_f, u = heapq.heappop(open_set)
        
        if u in closed_set:
            continue
        
        closed_set.add(u)
        nodes_visited_count += 1
        
        if u == target:
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, g_score[target], nodes_visited_count
        
        for v in G.successors(u):
            if v in closed_set:
                continue
            
            edge_data = G[u][v]
            edge_weight = edge_data.get(weight, float('inf'))
            
            if edge_weight < 0:
                error_msg = f"Negative weight found for edge ({u}->{v}): {weight}={edge_weight}. Weights must be non-negative for A* algorithm."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            tentative_g_score = g_score[u] + edge_weight
            
            if tentative_g_score < g_score[v]:
                came_from[v] = u
                g_score[v] = tentative_g_score
                f_score[v] = tentative_g_score + heuristic_func(v, target)
                heapq.heappush(open_set, (f_score[v], v))
    
    error_msg = f"Could not find a path from node {start} to node {target} using A* algorithm. The nodes may not be connected in the graph."
    logger.error(error_msg)
    raise nx.NetworkXNoPath(error_msg)


def route_shortest_a_star_by_coords(
    G: nx.DiGraph,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    use_manual: bool = True
) -> Dict:
    """
    Compute shortest-distance path using A*. Inputs are coordinates (lat, lon).
    Uses 'length' as the edge weight.
    
    Args:
        use_manual: If True, uses manual A* implementation with metrics tracking
    
    Returns a dict containing:
      - start_node, end_node, path_nodes, edges, street_segments (compressed),
        total_length_m, total_time_min, total_fuel_liters, execution_time_seconds,
        nodes_visited (if use_manual=True)
    """
    import time as time_module
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    heuristic = _make_astar_heuristic_shortest(G)
    
    start_time = time_module.perf_counter()
    try:
        if use_manual:
            path, _, nodes_visited = astar_manual_with_metrics(
                G, start_node, end_node, heuristic, weight="length"
            )
        else:
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="length")
            nodes_visited = None
    except nx.NetworkXNoPath:
        error_msg = f"Could not find a path between nodes (origin: {start_node}, destination: {end_node}) using A* shortest path algorithm. Please verify that the addresses are connected in the street network."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
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
    
    result = {
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
    
    if nodes_visited is not None:
        result["nodes_visited"] = nodes_visited
    
    return result


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


def _make_astar_heuristic_eco(G: nx.DiGraph) -> Callable[[int, int], float]:
    """
    Build an admissible heuristic for the ecological cost:
    - fuel lower bound: base fuel per meter * straight-line distance (assume no slope and no speed penalty)
    - time penalty lower bound: assume best-case speed (max observed in graph) to get minimum time,
      then convert to liters-equivalent with the same TIME_WEIGHT used in dijkstra preprocessing.
    
    This heuristic never overestimates (admissible) and therefore preserves A* optimality.
    """
    from calculation.dijkstra import BASE_L_PER_100KM, REF_SPEED_KMH, TIME_WEIGHT
    
    base_per_m = BASE_L_PER_100KM / 100000.0
    
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
    
    ref_speed_m_per_min = REF_SPEED_KMH * 1000.0 / 60.0
    liters_per_min_ref = (BASE_L_PER_100KM / 100000.0) * ref_speed_m_per_min
    
    def heuristic(u: int, v: int) -> float:
        lat_u = float(G.nodes[u].get("y", 0.0))
        lon_u = float(G.nodes[u].get("x", 0.0))
        lat_v = float(G.nodes[v].get("y", 0.0))
        lon_v = float(G.nodes[v].get("x", 0.0))
        dist = haversine(lon_u, lat_u, lon_v, lat_v)
        
        fuel_lb = base_per_m * dist
        
        time_min_lb = dist / max_speed_m_per_min if max_speed_m_per_min > 0 else 0.0
        time_penalty_lb = TIME_WEIGHT * time_min_lb * liters_per_min_ref
        
        return fuel_lb + time_penalty_lb
    
    return heuristic


def route_ecological_a_star_by_coords(
    G: nx.DiGraph,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    use_manual: bool = True
) -> Dict:
    """
    Compute eco-optimal path using A*. Inputs are coordinates (lat, lon).
    
    Args:
        use_manual: If True, uses manual A* implementation with metrics tracking
    
    Returns a dict containing:
      - start_node, end_node, path_nodes, edges, street_segments (compressed),
        total_length_m, total_time_min, total_fuel_liters, execution_time_seconds,
        nodes_visited (if use_manual=True)
    """
    import time as time_module
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    heuristic = _make_astar_heuristic_eco(G)
    
    start_time = time_module.perf_counter()
    try:
        if use_manual:
            path, _, nodes_visited = astar_manual_with_metrics(
                G, start_node, end_node, heuristic, weight="eco_cost"
            )
        else:
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="eco_cost")
            nodes_visited = None
    except nx.NetworkXNoPath:
        error_msg = f"Could not find a path between nodes (origin: {start_node}, destination: {end_node}) using A* ecological route algorithm. Please verify that the addresses are connected in the street network."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
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
    
    result = {
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
    
    if nodes_visited is not None:
        result["nodes_visited"] = nodes_visited
    
    return result


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


def calculate_astar_routes(start_addr: str, dest_addr: str) -> Tuple[Dict, Dict]:
    G = build_graph_from_csv()
    result_eco = route_ecological_a_star_by_addresses(G, start_addr, dest_addr)
    result_short = route_shortest_a_star_by_addresses(G, start_addr, dest_addr)
    return result_eco, result_short