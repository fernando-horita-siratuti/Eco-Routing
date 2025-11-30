import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import networkx as nx
import numpy as np
from geopy.geocoders import Nominatim
from calculation.elevation import street_steepness

# ========== PARÂMETROS ==========
BASE_L_PER_100KM = 10.0       # consumo base típico (L/100km) em velocidade moderada
SLOPE_COEF = 10.0             # quanto a subida aumenta o consumo (multiplicador por unidade de slope)
SPEED_PENALTY_COEF = 0.2      # penalidade por velocidades fora da referência (quadrática)
REF_SPEED_KMH = 50.0          # velocidade de referência para consumo (km/h)
TIME_WEIGHT = 0.5             # quantos "litros equivalentes" atribuímos a 1 minuto extra (fator multiplica)
# =========================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NODES_CSV = DEFAULT_DATA_DIR / "divinopolis_nodes.csv"
EDGES_CSV = DEFAULT_DATA_DIR / "divinopolis_edges.csv"


def _safe_float(val: object, fallback: float = 0.0) -> float:
    """
    Converte valor para float de forma segura, retornando fallback em caso de erro.
    
    Args:
        val: Valor a ser convertido
        fallback: Valor padrão caso a conversão falhe
    
    Returns:
        Valor convertido para float ou fallback
    """
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return fallback
        return float(val)
    except Exception:
        return fallback


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula a distância haversine entre dois pontos geográficos.
    
    Args:
        lon1: Longitude do primeiro ponto
        lat1: Latitude do primeiro ponto
        lon2: Longitude do segundo ponto
        lat2: Latitude do segundo ponto
    
    Returns:
        Distância em metros
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
    Parse do valor de maxspeed de uma string ou número.
    
    Args:
        val: Valor a ser parseado (string ou número)
        default: Valor padrão caso o parse falhe
    
    Returns:
        Velocidade máxima em km/h
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
    Constrói um grafo direcionado a partir dos arquivos CSV de nós e arestas.
    
    Args:
        nodes_csv: Caminho para o arquivo CSV de nós
        edges_csv: Caminho para o arquivo CSV de arestas
    
    Returns:
        Grafo direcionado NetworkX com todos os atributos calculados
    """
    if not nodes_csv.exists():
        raise FileNotFoundError(f"Nodes CSV não encontrado em: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"Edges CSV não encontrado em: {edges_csv}")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.DiGraph()

    # adiciona nós com lat/lon/elev
    for _, r in nodes_df.iterrows():
        nid = int(r['node_id'])
        lat = _safe_float(r.get('latitude'), fallback=0.0)
        lon = _safe_float(r.get('longitude'), fallback=0.0)
        elev_val = r.get('elevation', '')
        elevation = _safe_float(elev_val, fallback=0.0)
        G.add_node(nid, y=lat, x=lon, elevation=elevation)

    # adiciona arestas
    edges_invalid = 0
    for _, r in edges_df.iterrows():
        try:
            u = int(r['source_node'])
            v = int(r['target_node'])
        except Exception:
            continue
        
        length = _safe_float(r.get('length'), fallback=0.0)
        
        # Validação: ignora arestas com comprimento inválido desde o início
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
        print(f"Aviso: {edges_invalid} arestas com comprimento inválido foram ignoradas durante a leitura do CSV.")

    _precompute_edge_costs(G)
    
    # Validação final do grafo
    validate_graph_weights(G, weight_attr='eco_cost')
    validate_graph_weights(G, weight_attr='length')
    
    return G


def validate_graph_weights(G: nx.DiGraph, weight_attr: str = 'eco_cost') -> bool:
    """
    Valida que todas as arestas têm pesos válidos e positivos.
    Retorna True se válido, False caso contrário.
    """
    issues = []
    
    for u, v, data in G.edges(data=True):
        weight = data.get(weight_attr, None)
        
        if weight is None:
            issues.append(f"Aresta ({u}->{v}) não tem atributo '{weight_attr}'")
        elif math.isnan(weight):
            issues.append(f"Aresta ({u}->{v}) tem peso NaN")
        elif math.isinf(weight):
            issues.append(f"Aresta ({u}->{v}) tem peso infinito")
        elif weight < 0:
            issues.append(f"Aresta ({u}->{v}) tem peso negativo: {weight}")
    
    if issues:
        print(f"\nERRO: {len(issues)} problemas encontrados nos pesos do grafo (atributo '{weight_attr}'):")
        for issue in issues[:10]:  # Mostra apenas os primeiros 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... e mais {len(issues) - 10} problemas")
        return False
    
    return True


def _precompute_edge_costs(G: nx.DiGraph) -> None:
    """Calcula fuel_liters, time_minutes e eco_cost para cada aresta do grafo.
       Usa street_steepness para obter a grade (mais robusto que diferença/length simples).
       Garante que todos os custos sejam positivos e válidos."""
    base_per_m = BASE_L_PER_100KM / 100000.0  # L por metro
    ref_speed_kmh = REF_SPEED_KMH
    ref_speed_m_per_min = ref_speed_kmh * 1000.0 / 60.0
    liters_per_min_ref = base_per_m * ref_speed_m_per_min

    edges_removed = 0
    for u, v, data in list(G.edges(data=True)):
        length = float(data.get('length', 1.0))
        
        # Validação: remove arestas com comprimento inválido
        if length <= 0 or math.isnan(length) or math.isinf(length):
            G.remove_edge(u, v)
            edges_removed += 1
            continue
        
        speed_kmh = float(data.get('maxspeed_kmh', REF_SPEED_KMH))
        
        # Validação: velocidade deve ser positiva
        if speed_kmh <= 0 or math.isnan(speed_kmh) or math.isinf(speed_kmh):
            speed_kmh = REF_SPEED_KMH

        lat_u = float(G.nodes[u].get('y', 0.0))
        lon_u = float(G.nodes[u].get('x', 0.0))
        elev_u = float(G.nodes[u].get('elevation', 0.0))

        lat_v = float(G.nodes[v].get('y', 0.0))
        lon_v = float(G.nodes[v].get('x', 0.0))
        elev_v = float(G.nodes[v].get('elevation', 0.0))

        # usa street_steepness para obter grade (dh/dist_horizontal)
        try:
            steep = street_steepness(lat_u, lon_u, elev_u, lat_v, lon_v, elev_v)
            grade = steep.get("grade")
            # se grade for None (dist_h == 0), set 0
            slope = grade if grade is not None and not (math.isnan(grade) or math.isinf(grade)) else 0.0
        except Exception as e:
            print(f"Erro ao calcular steepness para aresta ({u}->{v}): {e}. Usando slope=0.")
            slope = 0.0
        
        uphill = max(slope, 0.0)

        # fatores
        slope_multiplier = 1.0 + (SLOPE_COEF * uphill)
        speed_factor = 1.0 + SPEED_PENALTY_COEF * ((speed_kmh - ref_speed_kmh) / ref_speed_kmh) ** 2

        fuel_liters = base_per_m * length * slope_multiplier * speed_factor

        speed_m_per_min = speed_kmh * 1000.0 / 60.0
        time_minutes = length / speed_m_per_min if speed_m_per_min > 0 else float('inf')

        time_penalty_equiv_liters = TIME_WEIGHT * time_minutes * liters_per_min_ref

        eco_cost = fuel_liters + time_penalty_equiv_liters
        
        # Validação crítica: garante que todos os custos sejam positivos e finitos
        if math.isnan(eco_cost) or math.isinf(eco_cost) or eco_cost < 0:
            # Se o custo for inválido, usa um valor mínimo seguro
            eco_cost = max(0.001, length * 0.00001)  # Custo mínimo baseado no comprimento
            print(f"Aviso: Aresta ({u}->{v}) tinha custo inválido. Corrigido para {eco_cost:.6f}")

        # Validação adicional para fuel_liters e time_minutes
        if math.isnan(fuel_liters) or math.isinf(fuel_liters) or fuel_liters < 0:
            fuel_liters = max(0.0, base_per_m * length)
        
        if math.isnan(time_minutes) or math.isinf(time_minutes) or time_minutes < 0:
            time_minutes = max(0.001, length / (ref_speed_m_per_min * 60)) if ref_speed_m_per_min > 0 else 0.001

        data['fuel_liters'] = fuel_liters
        data['time_minutes'] = time_minutes
        data['eco_cost'] = eco_cost
        data['slope'] = slope
    
    if edges_removed > 0:
        print(f"Aviso: {edges_removed} arestas com comprimento inválido foram removidas.")


def nearest_node_to_point(G: nx.DiGraph, lat: float, lon: float) -> int:
    """Busca o nó mais próximo por distância haversine (simples e robusto para cidade)."""
    nodes = list(G.nodes(data=True))
    coords = np.array([[n[1]['y'], n[1]['x']] for n in nodes])
    lat_arr = coords[:, 0].astype(float)
    lon_arr = coords[:, 1].astype(float)
    dists = np.array([haversine(lon, lat, lon_arr[i], lat_arr[i]) for i in range(len(lat_arr))])
    idx = int(np.argmin(dists))
    nearest_node = nodes[idx][0]
    return nearest_node

def geocode_address(address: str, user_agent: str = "meu_app", timeout: int = 10) -> Tuple[float, float, str]:
    """
    Faz geocoding de um endereço com retry e tratamento de erros melhorado.
    
    Args:
        address: Endereço a ser geocodificado
        user_agent: User agent para o Nominatim
        timeout: Timeout em segundos
    
    Returns:
        Tuple (latitude, longitude, endereço encontrado)
    """
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    import time
    
    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    
    # Tentativas com variações do endereço
    variations = [
        address,  # Tenta o endereço original primeiro
        address.replace(",", ""),  # Remove vírgulas
        address.split(",")[0] + ", Divinópolis, MG, Brasil",  # Só rua e cidade
        address.split(",")[0] + ", Divinópolis, Brasil",  # Sem estado
    ]
    
    last_error = None
    for i, addr_variant in enumerate(variations):
        try:
            loc = geolocator.geocode(addr_variant, timeout=timeout)
            
            if loc is not None:
                return loc.latitude, loc.longitude, loc.address
            
            # Aguarda um pouco antes da próxima tentativa
            if i < len(variations) - 1:
                time.sleep(1)
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            last_error = e
            if i < len(variations) - 1:
                time.sleep(2)  # Aguarda mais em caso de erro
    
    # Se todas as tentativas falharam
    error_msg = f"Geocoding falhou para: {address}"
    if last_error:
        error_msg += f" (último erro: {last_error})"
    
    raise ValueError(error_msg)


def _select_best_edge_between(G: nx.DiGraph, u: int, v: int) -> Optional[Dict]:
    """Se MultiGraph escolhe aresta com menor eco_cost; se DiGraph simples devolve atributos."""
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
    """Agrega segmentos consecutivos com o mesmo nome."""
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
    Calcula a rota ecológica (otimizada para menor consumo de combustível).
    
    Args:
        G: Grafo direcionado
        start_addr: Endereço de origem
        dest_addr: Endereço de destino
    
    Returns:
        Dicionário com informações da rota (path_nodes, edges, métricas, tempo de execução)
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)

    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)

    # Mede tempo de execução do algoritmo
    start_time = time_module.perf_counter()
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='eco_cost', method='dijkstra')
    except nx.NetworkXNoPath:
        raise RuntimeError("Não há caminho entre os nós selecionados.")
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
    Implementação manual do algoritmo de Dijkstra.
    Retorna (caminho, custo_total).
    
    Args:
        G: Grafo direcionado
        start: Nó de origem
        target: Nó de destino
        weight: Atributo da aresta a ser usado como peso ('length', 'eco_cost', etc.)
    """
    import heapq
    
    # Inicialização
    dist = {node: float('inf') for node in G.nodes()}
    dist[start] = 0.0
    prev = {node: None for node in G.nodes()}
    visited = set()
    
    # Fila de prioridade: (distância, nó)
    pq = [(0.0, start)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
            
        visited.add(u)
        
        # Se chegamos no destino, podemos parar
        if u == target:
            break
        
        # Explora vizinhos
        for v in G.successors(u):
            if v in visited:
                continue
                
            edge_data = G[u][v]
            edge_weight = edge_data.get(weight, float('inf'))
            
            if edge_weight < 0:
                raise ValueError(f"Peso negativo encontrado: {weight}={edge_weight}")
            
            alt = current_dist + edge_weight
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    
    # Reconstrói o caminho
    if dist[target] == float('inf'):
        raise nx.NetworkXNoPath(f"Não há caminho de {start} para {target}")
    
    path = []
    u = target
    while u is not None:
        path.append(u)
        u = prev[u]
    path.reverse()
    
    return path, dist[target]


def _process_path(G: nx.DiGraph, path: List[int]) -> Dict:
    """Processa um caminho e calcula estatísticas (distância, combustível, tempo)."""
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
    Calcula a rota com menor distância (usa 'length' como peso).
    
    Args:
        G: Grafo
        start_addr: Endereço de origem
        dest_addr: Endereço de destino
        use_manual_dijkstra: Se True, usa implementação manual do Dijkstra
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)
    
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    # Mede tempo de execução do algoritmo
    start_time = time_module.perf_counter()
    try:
        if use_manual_dijkstra:
            path, _ = dijkstra_manual(G, start_node, end_node, weight='length')
        else:
            path = nx.shortest_path(G, source=start_node, target=end_node, weight='length', method='dijkstra')
    except nx.NetworkXNoPath:
        raise RuntimeError("Não há caminho entre os nós selecionados.")
    execution_time = time_module.perf_counter() - start_time
    
    result = _process_path(G, path)
    result['start_node'] = start_node
    result['end_node'] = end_node
    result['execution_time_seconds'] = execution_time
    
    return result


def route_ecological_manual_dijkstra(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    """
    Calcula rota ecológica usando implementação manual do Dijkstra.
    """
    import time as time_module
    start_lat, start_lon, _ = geocode_address(start_addr)
    dest_lat, dest_lon, _ = geocode_address(dest_addr)
    
    start_node = nearest_node_to_point(G, start_lat, start_lon)
    end_node = nearest_node_to_point(G, dest_lat, dest_lon)
    
    # Mede tempo de execução do algoritmo
    start_time = time_module.perf_counter()
    try:
        path, _ = dijkstra_manual(G, start_node, end_node, weight='eco_cost')
    except nx.NetworkXNoPath:
        raise RuntimeError("Não há caminho entre os nós selecionados.")
    execution_time = time_module.perf_counter() - start_time
    
    result = _process_path(G, path)
    result['start_node'] = start_node
    result['end_node'] = end_node
    result['execution_time_seconds'] = execution_time
    
    return result


def compare_routes(G: nx.DiGraph, start_addr: str, dest_addr: str) -> Dict:
    """
    Compara rota ecológica vs rota mais curta.
    Retorna dicionário com ambas as rotas e estatísticas comparativas.
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
    Calcula rotas e compara se solicitado.
    
    Args:
        compare: Se True, compara rota ecológica vs rota mais curta
        use_manual_dijkstra: Se True, usa implementação manual do Dijkstra para rota mais curta
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
    Calcula rotas usando Dijkstra (mais curta e ecológica).
    
    Args:
        start_addr: Endereço de origem
        dest_addr: Endereço de destino
    
    Returns:
        Tupla (result_short, result_eco) com os resultados das rotas
    """
    G = build_graph_from_csv()
    result_short = route_shortest_distance(G, start_addr, dest_addr)
    result_eco = route_ecological_manual_dijkstra(G, start_addr, dest_addr)
    return result_short, result_eco

