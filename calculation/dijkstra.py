import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import networkx as nx
import numpy as np
from geopy.geocoders import Nominatim
from calculation.elevation import street_steepness

# ========== PARAMETERS ==========
# TO ADAPT FOR ANOTHER CONTEXT: These parameters can be adjusted based on vehicle characteristics and local conditions
BASE_L_PER_100KM = 10.0       # typical base consumption (L/100km) at moderate speed
SLOPE_COEF = 10.0             # how much uphill increases consumption (multiplier per unit of slope)
SPEED_PENALTY_COEF = 0.2      # penalty for speeds outside reference (quadratic)
REF_SPEED_KMH = 50.0          # reference speed for consumption (km/h)
TIME_WEIGHT = 0.5             # how many "equivalent liters" we assign to 1 extra minute (multiplicative factor)
# =========================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# TO ADAPT FOR ANOTHER CITY: Change these CSV filenames to match your city
NODES_CSV = DEFAULT_DATA_DIR / "divinopolis_nodes.csv"
EDGES_CSV = DEFAULT_DATA_DIR / "divinopolis_edges.csv"


def _safe_float(val: object, fallback: float = 0.0) -> float:
    """
    Safely converts a value to float, returning fallback in case of error.
    
    Args:
        val: Value to be converted
        fallback: Default value if conversion fails
    
    Returns:
        Value converted to float or fallback
    """
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return fallback
        return float(val)
    except Exception:
        return fallback


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculates the haversine distance between two geographic points.
    
    Args:
        lon1: Longitude of the first point
        lat1: Latitude of the first point
        lon2: Longitude of the second point
        lat2: Latitude of the second point
    
    Returns:
        Distance in meters
    """
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


def parse_maxspeed(val, default=REF_SPEED_KMH):
    """
    Parses the maxspeed value from a string or number.
    
    Args:
        val: Value to be parsed (string or number)
        default: Default value if parsing fails
    
    Returns:
        Maximum speed in km/h
    """
    if pd.isna(val) or val == "":
        return default
    try:
        if isinstance(val, str):
            first = val.split(';')[0].strip()
            digits = ''.join(ch for ch in first if (ch.isdigit() or ch=='.'))
            return float(digits) if digits != "" else default
        else:
            return float(val)
    except:
        return default


def build_graph_from_csv(nodes_csv: Path = NODES_CSV, edges_csv: Path = EDGES_CSV) -> nx.DiGraph:
    """
    Builds a directed graph from CSV files of nodes and edges.
    
    Args:
        nodes_csv: Path to the nodes CSV file
        edges_csv: Path to the edges CSV file
    
    Returns:
        Directed NetworkX graph with all calculated attributes
    """
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Nodes CSV not found at: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"Edges CSV not found at: {edges_csv}")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.DiGraph()

    # Add nodes with lat/lon/elevation
    for _, r in nodes_df.iterrows():
        nid = int(r['node_id'])
        lat = _safe_float(r.get('latitude'), fallback=0.0)
        lon = _safe_float(r.get('longitude'), fallback=0.0)
        elev_val = r.get('elevation', '')
        elevation = _safe_float(elev_val, fallback=0.0)
        G.add_node(nid, y=lat, x=lon, elevation=elevation)

    # Add edges
    edges_invalid = 0
    for _, r in edges_df.iterrows():
        try:
            u = int(r['source_node'])
            v = int(r['target_node'])
        except Exception:
            continue
        
        length = _safe_float(r.get('length'), fallback=0.0)
        
        # Validation: ignore edges with invalid length from the start
        if length <= 0 or math.isnan(length) or math.isinf(length):
            edges_invalid += 1
            continue
        
        name = r.get('name', "") if pd.notna(r.get('name', "")) else ""
        maxspeed = parse_maxspeed(r.get('maxspeed', REF_SPEED_KMH), default=REF_SPEED_KMH)
        oneway = str(r.get('oneway', 'False')).lower() in ('true', '1', 't', 'yes')

        # ignore edges whose nodes are missing
        if u not in G.nodes or v not in G.nodes:
            continue

        G.add_edge(u, v, length=length, name=name, maxspeed_kmh=maxspeed, original=True)
        if not oneway:
            G.add_edge(v, u, length=length, name=name, maxspeed_kmh=maxspeed, original=True)

    if edges_invalid > 0:
        print(f"Warning: {edges_invalid} edges with invalid length were ignored during CSV reading.")

    _precompute_edge_costs(G)
    
    # Final graph validation
    validate_graph_weights(G, weight_attr='eco_cost')
    validate_graph_weights(G, weight_attr='length')
    
    return G


def validate_graph_weights(G: nx.DiGraph, weight_attr: str = 'eco_cost') -> bool:
    """
    Validates that all edges have valid and positive weights.
    Returns True if valid, False otherwise.
    """
    issues = []
    
    for u, v, data in G.edges(data=True):
        weight = data.get(weight_attr, None)
        
        if weight is None:
            issues.append(f"Edge ({u}->{v}) does not have attribute '{weight_attr}'")
        elif math.isnan(weight):
            issues.append(f"Edge ({u}->{v}) has NaN weight")
        elif math.isinf(weight):
            issues.append(f"Edge ({u}->{v}) has infinite weight")
        elif weight < 0:
            issues.append(f"Edge ({u}->{v}) has negative weight: {weight}")
    
    if issues:
        print(f"\nERROR: {len(issues)} problems found in graph weights (attribute '{weight_attr}'):")
        for issue in issues[:10]:  # Show only the first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more problems")
        return False
    
    return True


def _precompute_edge_costs(G: nx.DiGraph) -> None:
    """Calculates fuel_liters, time_minutes and eco_cost for each edge in the graph.
       Uses street_steepness to obtain the grade (more robust than simple difference/length).
       Ensures that all costs are positive and valid."""
    base_per_m = BASE_L_PER_100KM / 100000.0  # Liters per meter
    ref_speed_kmh = REF_SPEED_KMH
    ref_speed_m_per_min = ref_speed_kmh * 1000.0 / 60.0
    liters_per_min_ref = base_per_m * ref_speed_m_per_min

    edges_removed = 0
    for u, v, data in list(G.edges(data=True)):
        length = float(data.get('length', 1.0))
        
        # Validation: remove edges with invalid length
        if length <= 0 or math.isnan(length) or math.isinf(length):
            G.remove_edge(u, v)
            edges_removed += 1
            continue
        
        speed_kmh = float(data.get('maxspeed_kmh', REF_SPEED_KMH))
        
        # Validation: speed must be positive
        if speed_kmh <= 0 or math.isnan(speed_kmh) or math.isinf(speed_kmh):
            speed_kmh = REF_SPEED_KMH

        lat_u = float(G.nodes[u].get('y', 0.0))
        lon_u = float(G.nodes[u].get('x', 0.0))
        elev_u = float(G.nodes[u].get('elevation', 0.0))

        lat_v = float(G.nodes[v].get('y', 0.0))
        lon_v = float(G.nodes[v].get('x', 0.0))
        elev_v = float(G.nodes[v].get('elevation', 0.0))

        # Use street_steepness to obtain grade (dh/dist_horizontal)
        try:
            steep = street_steepness(lat_u, lon_u, elev_u, lat_v, lon_v, elev_v)
            grade = steep.get("grade")
            # If grade is None (dist_h == 0), set to 0
            slope = grade if grade is not None and not (math.isnan(grade) or math.isinf(grade)) else 0.0
        except Exception as e:
            print(f"Error calculating steepness for edge ({u}->{v}): {e}. Using slope=0.")
            slope = 0.0
        
        uphill = max(slope, 0.0)

        # Factors
        slope_multiplier = 1.0 + (SLOPE_COEF * uphill)
        speed_factor = 1.0 + SPEED_PENALTY_COEF * ((speed_kmh - ref_speed_kmh) / ref_speed_kmh) ** 2

        fuel_liters = base_per_m * length * slope_multiplier * speed_factor

        speed_m_per_min = speed_kmh * 1000.0 / 60.0
        time_minutes = length / speed_m_per_min if speed_m_per_min > 0 else float('inf')

        time_penalty_equiv_liters = TIME_WEIGHT * time_minutes * liters_per_min_ref

        eco_cost = fuel_liters + time_penalty_equiv_liters
        
        # Critical validation: ensures all costs are positive and finite
        if math.isnan(eco_cost) or math.isinf(eco_cost) or eco_cost < 0:
            # If cost is invalid, use a safe minimum value
            eco_cost = max(0.001, length * 0.00001)  # Minimum cost based on length
            print(f"Warning: Edge ({u}->{v}) had invalid cost. Corrected to {eco_cost:.6f}")

        # Additional validation for fuel_liters and time_minutes
        if math.isnan(fuel_liters) or math.isinf(fuel_liters) or fuel_liters < 0:
            fuel_liters = max(0.0, base_per_m * length)
        
        if math.isnan(time_minutes) or math.isinf(time_minutes) or time_minutes < 0:
            time_minutes = max(0.001, length / (ref_speed_m_per_min * 60)) if ref_speed_m_per_min > 0 else 0.001

        data['fuel_liters'] = fuel_liters
        data['time_minutes'] = time_minutes
        data['eco_cost'] = eco_cost
        data['slope'] = slope
    
    if edges_removed > 0:
        print(f"Warning: {edges_removed} edges with invalid length were removed.")


def nearest_node_to_point(G: nx.DiGraph, lat: float, lon: float) -> int:
    """Finds the nearest node by haversine distance (simple and robust for city-scale)."""
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes])
    lat_arr = coords[:, 0].astype(float)
    lon_arr = coords[:, 1].astype(float)
    dists = np.array([haversine(lon, lat, lon_arr[i], lat_arr[i]) for i in range(len(lat_arr))])
    idx = int(np.argmin(dists))
    nearest_node = nodes[idx][0]
    return nearest_node

def geocode_address(address: str, user_agent: str = "meu_app", timeout: int = 5) -> Tuple[float, float, str]:
    """
    Performs geocoding of an address with improved retry and error handling.
    Includes known neighborhoods of Divinópolis to improve accuracy.
    Optimized for speed while maintaining precision.
    
    NOTE: To adapt this for another city, modify:
    - bairros_divinopolis: Replace with neighborhoods/districts of your target city
    - street_to_bairro: Update with streets that have multiple locations in your city
    - Replace "Divinópolis, MG, Brasil" with your city's location string
    - Adjust regex patterns if your language uses different street prefixes
    
    Args:
        address: Address to be geocoded
        user_agent: User agent for Nominatim
        timeout: Timeout in seconds (reduced to 5)
    
    Returns:
        Tuple (latitude, longitude, found address)
    """
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    import time
    import re
    
    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    
    # Known neighborhoods of Divinópolis to improve search accuracy
    # TO ADAPT FOR ANOTHER CITY: Replace this list with neighborhoods/districts of your target city
    bairros_divinopolis = [
        "Centro", "Esplanada", "Niterói", "Candidés", "Vila Romana", 
        "Jardim Belvedere", "Jardim Paraíso", "Santo Antônio", "Bela Vista",
        "Morro da Pitimba", "Orion", "Itaí", "Danilo Passos", "Chácaras",
        "Jardim Copacabana", "Vila Rica", "São Luiz", "Vila Dom Bosco",
        "São Sebastião", "Santa Rosa"  # Added common neighborhoods that may have duplicate street names
    ]
    
    # Normalize the address
    address_clean = ", ".join([part.strip() for part in address.split(",") if part.strip()])
    address_lower = address_clean.lower()
    
    # Extract address information
    # TO ADAPT FOR ANOTHER LANGUAGE: Modify regex patterns to match your language's street prefixes
    street_match = re.search(r'(rua|avenida|av\.?|r\.?)\s+([^,0-9]+?)(?:\s*,\s*|\s*$)', address_lower)
    if not street_match:
        street_match = re.search(r'^([^,0-9]+?)(?:\s*,\s*\d)', address_lower)
    
    number_match = re.search(r'(\d+)', address_clean)
    
    street_name = street_match.group(2).strip() if street_match else None
    if not street_name and street_match:
        street_name = street_match.group(1).strip() if street_match.lastindex >= 1 else None
    
    street_number = number_match.group(1) if number_match else None
    
    # Mapping of known streets to their most likely neighborhoods
    # Useful for streets that have multiple locations in the same city
    # TO ADAPT FOR ANOTHER CITY: Update this dictionary with streets that have duplicates in your city
    street_to_bairro = {
        "goiás": ["Orion", "Centro", "Esplanada"],
        "goias": ["Orion", "Centro", "Esplanada"],
        "pains": ["Centro", "Esplanada", "Niterói"],
        "bahia": ["Centro", "São Sebastião", "Esplanada"],  # Rua Bahia exists in Centro and São Sebastião
    }
    
    # Check if address already mentions a neighborhood
    mentioned_bairro = None
    mentioned_bairro_lower = None
    for bairro in bairros_divinopolis:
        bairro_lower = bairro.lower()
        if bairro_lower in address_lower:
            mentioned_bairro = bairro
            mentioned_bairro_lower = bairro_lower
            break
    
    # OPTIMIZATION: Build only the most important variations (priority)
    variations = []
    
    # 1. MAXIMUM PRIORITY: If neighborhood mentioned, create variation with neighborhood FIRST
    # TO ADAPT FOR ANOTHER CITY: Replace "Divinópolis, MG, Brasil" with your city's location string
    if mentioned_bairro and street_name and street_number:
        street_prefix = street_match.group(1) if street_match and street_match.lastindex >= 1 else "Rua"
        street_full = f"{street_prefix.title()} {street_name.title()}, {street_number}"
        variations.append(f"{street_full}, {mentioned_bairro}, Divinópolis, MG, Brasil")
    
    # 2. Complete original address (second priority)
    if "divinópolis" in address_lower or "divinopolis" in address_lower:
        variations.append(address_clean)
    
    # 3. If we have street name and number, try only with 2-3 most likely neighborhoods
    if street_name and street_number:
        street_prefix = street_match.group(1) if street_match and street_match.lastindex >= 1 else "Rua"
        street_full = f"{street_prefix.title()} {street_name.title()}, {street_number}"
        
        if mentioned_bairro:
            # If already added above, don't add again
            pass
        else:
            # OPTIMIZATION: Only 2-3 most likely neighborhoods
            street_key = street_name.lower().strip()
            preferred_bairros = street_to_bairro.get(street_key, ["Centro", "Esplanada", "Niterói"])
            # Limit to 3 neighborhoods
            for bairro in preferred_bairros[:3]:
                variations.append(f"{street_full}, {bairro}, Divinópolis, MG, Brasil")
    
    # 4. Variations without neighborhood (only if not found yet)
    if street_name:
        street_prefix = street_match.group(1) if street_match and street_match.lastindex >= 1 else "Rua"
        street_full = f"{street_prefix.title()} {street_name.title()}"
        if street_number:
            variations.append(f"{street_full}, {street_number}, Divinópolis, MG, Brasil")
        variations.append(f"{street_full}, Divinópolis, MG, Brasil")
    
    # 5. Minimum fallback
    if not variations:
        variations.append(address_clean)
        if "divinópolis" not in address_lower:
            variations.append(f"{address_clean}, Divinópolis, MG, Brasil")
    
    # Remove duplicates maintaining order (first occurrence)
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    # OPTIMIZATION: Limit to maximum 6 attempts (prioritizes the most important)
    max_attempts = min(6, len(unique_variations))
    unique_variations = unique_variations[:max_attempts]
    
    last_error = None
    best_result = None
    best_score = 0
    
    for i, addr_variant in enumerate(unique_variations):
        try:
            loc = geolocator.geocode(
                addr_variant,
                timeout=timeout,
                addressdetails=True,
                exactly_one=True
            )
            
            if loc is not None:
                # Validate the result
                returned_address_lower = loc.address.lower()
                score = 0
                
                # Scoring based on matches
                if street_name:
                    street_lower = street_name.lower().strip()
                    if street_lower in returned_address_lower:
                        score += 10
                    elif street_lower.replace(" ", "") in returned_address_lower.replace(" ", ""):
                        score += 8
                
                if street_number and street_number in returned_address_lower:
                    score += 20
                if "divinópolis" in returned_address_lower or "divinopolis" in returned_address_lower:
                    score += 5
                
                # CRITICAL: Check mentioned neighborhood vs returned neighborhood
                # This solves the problem of duplicate streets
                if mentioned_bairro_lower:
                    if mentioned_bairro_lower in returned_address_lower:
                        # HIGH BONUS: Mentioned neighborhood matches returned one
                        score += 25  # Very high bonus for exact match
                    else:
                        # PENALTY: Returned neighborhood is different from mentioned one
                        # Check if any other known neighborhood appears in the result
                        other_bairro_found = False
                        for bairro in bairros_divinopolis:
                            if bairro.lower() != mentioned_bairro_lower and bairro.lower() in returned_address_lower:
                                other_bairro_found = True
                                break
                        if other_bairro_found:
                            score -= 15  # Significant penalty for different neighborhood
                
                # Bonus if in correct neighborhood (based on mapping)
                if street_name:
                    street_key = street_name.lower().strip()
                    preferred_bairros = street_to_bairro.get(street_key, [])
                    for bairro in preferred_bairros:
                        if bairro.lower() in returned_address_lower:
                            score += 10
                            break
                
                # OPTIMIZATION: Stop immediately if found very good result
                # Increased to 30 to ensure neighborhood match is sufficient
                if score >= 30:
                    return loc.latitude, loc.longitude, loc.address
                
                # Keep the best result
                if score > best_score:
                    best_score = score
                    best_result = (loc.latitude, loc.longitude, loc.address)
            
            # OPTIMIZATION: Minimum wait between attempts
            if i < len(unique_variations) - 1:
                # If already has very good result, stop
                if best_score >= 30:
                    break
                # If already has reasonable result, wait less
                elif best_score >= 20:
                    time.sleep(0.2)
                else:
                    time.sleep(0.3)
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            last_error = e
            # If already has reasonable result, don't continue
            if best_score >= 20:
                break
            if i < len(unique_variations) - 1:
                time.sleep(1)
    
    # If found any result, return the best one
    if best_result is not None:
        return best_result
    
    # If all attempts failed
    error_msg = f"Geocoding failed for: {address}"
    if last_error:
        error_msg += f" (last error: {last_error})"
    
    raise ValueError(error_msg)


def _select_best_edge_between(G: nx.DiGraph, u: int, v: int) -> Optional[Dict]:
    """If MultiGraph, chooses edge with lowest eco_cost; if simple DiGraph, returns attributes."""
    if G.is_multigraph():
        ed = G.get_edge_data(u, v)
        if not ed:
            return None
        best_data = None
        best_cost = float('inf')
        for k, attr in ed.items():
            cost = attr.get('eco_cost', float('inf'))
            if cost < best_cost:
                best_cost = cost
                best_data = attr
        return best_data
    else:
        return G[u][v] if G.has_edge(u, v) else None


def compress_street_segments(segments: List[Tuple[str, float, float, float]]) -> List[Tuple[str, float, float, float]]:
    """Aggregates consecutive segments with the same name."""
    if not segments:
        return []
    out = []
    cur_name, cur_len, cur_fuel, cur_time = segments[0]
    for name, length, fuel, time in segments[1:]:
        if name == cur_name:
            cur_len += length
            cur_fuel += fuel
            cur_time += time
        else:
            out.append((cur_name, cur_len, cur_fuel, cur_time))
            cur_name, cur_len, cur_fuel, cur_time = name, length, fuel, time
    out.append((cur_name, cur_len, cur_fuel, cur_time))
    return out


def route_ecological(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    """
    Calculates the ecological route (optimized for lowest fuel consumption).
    
    Args:
        G: Directed graph
        start_addr: Origin address
        dest_addr: Destination address
    
    Returns:
        Dictionary with route information (path_nodes, edges, metrics, execution time)
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)

    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)

    # Measure algorithm execution time
    start_time = time_module.perf_counter()
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='eco_cost', method='dijkstra')
    except nx.NetworkXNoPath:
        raise RuntimeError("No path between selected nodes.")
    execution_time = time_module.perf_counter() - start_time

    total_length = 0.0
    total_fuel = 0.0
    total_time_min = 0.0
    edges = []
    street_segments = []

    for i in range(len(path) - 1):
        u = path[i]; v = path[i + 1]
        data = _select_best_edge_between(G, u, v)
        if data is None:
            continue
        length = data.get('length', 0.0)
        fuel = data.get('fuel_liters', 0.0)
        time_min = data.get('time_minutes', 0.0)
        name = data.get('name') if data.get('name') else "unnamed"
        total_length += length
        total_fuel += fuel
        total_time_min += time_min
        edges.append((u, v, data))
        street_segments.append((name, length, fuel, time_min))

    street_segments_compressed = compress_street_segments(street_segments)

    return {
        'start_node': start_node,
        'end_node': end_node,
        'path_nodes': path,
        'edges': edges,
        'street_segments': street_segments_compressed,
        'total_length_m': total_length,
        'total_time_min': total_time_min,
        'total_fuel_liters': total_fuel,
        'execution_time_seconds': execution_time
    }


def dijkstra_manual(G: nx.DiGraph, start: int, target: int, weight: str = 'length') -> Tuple[List[int], float]:
    """
    Manual implementation of Dijkstra's algorithm.
    Returns (path, total_cost).
    
    Args:
        G: Directed graph
        start: Origin node
        target: Destination node
        weight: Edge attribute to use as weight ('length', 'eco_cost', etc.)
    """
    import heapq
    
    # Initialization
    dist = {node: float('inf') for node in G.nodes()}
    dist[start] = 0.0
    prev = {node: None for node in G.nodes()}
    visited = set()
    
    # Priority queue: (distance, node)
    pq = [(0.0, start)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
            
        visited.add(u)
        
        # If we reached the destination, we can stop
        if u == target:
            break
        
        # Explore neighbors
        for v in G.successors(u):
            if v in visited:
                continue
                
            edge_data = G[u][v]
            edge_weight = edge_data.get(weight, float('inf'))
            
            if edge_weight < 0:
                raise ValueError(f"Negative weight found: {weight}={edge_weight}")
            
            alt = current_dist + edge_weight
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    
    # Reconstruct the path
    if dist[target] == float('inf'):
        raise nx.NetworkXNoPath(f"No path from {start} to {target}")
    
    path = []
    u = target
    while u is not None:
        path.append(u)
        u = prev[u]
    path.reverse()
    
    return path, dist[target]


def _process_path(G: nx.DiGraph, path: List[int]) -> Dict:
    """Processes a path and calculates statistics (distance, fuel, time)."""
    total_length = 0.0
    total_fuel = 0.0
    total_time_min = 0.0
    edges = []
    street_segments = []
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        data = _select_best_edge_between(G, u, v)
        if data is None:
            continue
        length = data.get('length', 0.0)
        fuel = data.get('fuel_liters', 0.0)
        time_min = data.get('time_minutes', 0.0)
        name = data.get('name') if data.get('name') else "unnamed"
        total_length += length
        total_fuel += fuel
        total_time_min += time_min
        edges.append((u, v, data))
        street_segments.append((name, length, fuel, time_min))
    
    street_segments_compressed = compress_street_segments(street_segments)
    
    return {
        'path_nodes': path,
        'edges': edges,
        'street_segments': street_segments_compressed,
        'total_length_m': total_length,
        'total_time_min': total_time_min,
        'total_fuel_liters': total_fuel
    }


def route_shortest_distance(G: nx.DiGraph, start_addr: str, dest_addr: str, use_manual_dijkstra: bool = False) -> Dict:
    """
    Calculates the route with shortest distance (uses 'length' as weight).
    
    Args:
        G: Graph
        start_addr: Origin address
        dest_addr: Destination address
        use_manual_dijkstra: If True, uses manual Dijkstra implementation
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)
    
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    # Measure algorithm execution time
    start_time = time_module.perf_counter()
    try:
        if use_manual_dijkstra:
            path, _ = dijkstra_manual(G, start_node, end_node, weight='length')
        else:
            path = nx.shortest_path(G, source=start_node, target=end_node, weight='length', method='dijkstra')
    except nx.NetworkXNoPath:
        raise RuntimeError("No path between selected nodes.")
    execution_time = time_module.perf_counter() - start_time
    
    result = _process_path(G, path)
    result['start_node'] = start_node
    result['end_node'] = end_node
    result['execution_time_seconds'] = execution_time
    
    return result


def route_ecological_manual_dijkstra(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    """
    Calculates ecological route using manual Dijkstra implementation.
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)
    
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    # Measure algorithm execution time
    start_time = time_module.perf_counter()
    try:
        path, _ = dijkstra_manual(G, start_node, end_node, weight='eco_cost')
    except nx.NetworkXNoPath:
        raise RuntimeError("No path between selected nodes.")
    execution_time = time_module.perf_counter() - start_time
    
    result = _process_path(G, path)
    result['start_node'] = start_node
    result['end_node'] = end_node
    result['execution_time_seconds'] = execution_time
    
    return result


def compare_routes(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    """
    Compares ecological route vs shortest route.
    Returns dictionary with both routes and comparative statistics.
    """
    import time as time_module
    dijkstra_start_time = time_module.perf_counter()
    
    route_eco = route_ecological(G, start_addr, dest_addr)
    route_short = route_shortest_distance(G, start_addr, dest_addr)
    
    dijkstra_total_time = time_module.perf_counter() - dijkstra_start_time
    
    return {
        'ecological': route_eco,
        'shortest': route_short,
        'comparison': {
            'length_diff_m': route_eco['total_length_m'] - route_short['total_length_m'],
            'length_diff_pct': ((route_eco['total_length_m'] - route_short['total_length_m']) / route_short['total_length_m']) * 100 if route_short['total_length_m'] > 0 else 0,
            'fuel_diff_liters': route_short['total_fuel_liters'] - route_eco['total_fuel_liters'],
            'fuel_diff_pct': ((route_eco['total_fuel_liters'] - route_short['total_fuel_liters']) / route_short['total_fuel_liters']) * 100 if route_short['total_fuel_liters'] > 0 else 0,
            'time_diff_min': route_eco['total_time_min'] - route_short['total_time_min'],
            'time_diff_pct': ((route_eco['total_time_min'] - route_short['total_time_min']) / route_short['total_time_min']) * 100 if route_short['total_time_min'] > 0 else 0,
        },
        'total_execution_time_seconds': dijkstra_total_time
    }


def calculate_route(compare: bool = True, use_manual_dijkstra: bool = False):
    """
    Calculates routes and compares if requested.
    
    Args:
        compare: If True, compares ecological route vs shortest route
        use_manual_dijkstra: If True, uses manual Dijkstra implementation for shortest route
    """
    G = build_graph_from_csv()
    
    start_address = "Rua Padre Eustáquio, 716, Divinópolis, MG, Brasil"
    dest_address = "Rua Rio de Janeiro, 2220, Divinópolis, MG, Brasil"
    
    if compare:
        results = compare_routes(G, start_address, dest_address)
        return results
    else:
        result = route_ecological(G, start_address, dest_address)
        
        return result

def calculate_route_dijkstra(start_addr: str, dest_addr: str):
    """
    Calculates routes using Dijkstra (shortest and ecological).
    
    Args:
        start_addr: Origin address
        dest_addr: Destination address
    
    Returns:
        Tupla (result_short, result_eco) com os resultados das rotas
    """
    G = build_graph_from_csv()
    result_short = route_shortest_distance(G, start_addr, dest_addr)
    result_eco = route_ecological_manual_dijkstra(G, start_addr, dest_addr)
    return result_short, result_eco

