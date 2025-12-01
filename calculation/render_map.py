from pathlib import Path
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import re
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from .dijkstra import build_graph_from_csv, calculate_route_dijkstra
from .a_star import calculate_astar_routes

def reverse_geocode(lat: float, lon: float, user_agent: str = "meu_app", timeout: int = 5) -> str:
    """
    Faz reverse geocoding de coordenadas para obter endereço.
    
    Args:
        lat: Latitude
        lon: Longitude
        user_agent: User agent para o Nominatim
        timeout: Timeout em segundos
    
    Returns:
        Endereço encontrado ou coordenadas como fallback
    """
    try:
        geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        location = geolocator.reverse((lat, lon), timeout=timeout)
        if location and location.address:
            return location.address
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Erro no reverse geocoding para ({lat}, {lon}): {e}")
    except Exception as e:
        print(f"Erro inesperado no reverse geocoding: {e}")
    
    # Fallback: retorna coordenadas formatadas
    return f"Lat: {lat:.6f}, Lon: {lon:.6f}"


def get_key_points(G, path_nodes):
    """
    Seleciona apenas os pontos de início e fim da rota.
    
    Args:
        G: Grafo
        path_nodes: Lista de nós do caminho
    
    Returns:
        Lista de tuplas (node_id, lat, lon) com apenas início e fim
    """
    if len(path_nodes) < 2:
        return [(path_nodes[0], float(G.nodes[path_nodes[0]]['y']), float(G.nodes[path_nodes[0]]['x']))]
    
    return [
        (path_nodes[0], float(G.nodes[path_nodes[0]]['y']), float(G.nodes[path_nodes[0]]['x'])),
        (path_nodes[-1], float(G.nodes[path_nodes[-1]]['y']), float(G.nodes[path_nodes[-1]]['x']))
    ]


def create_single_metric_chart(result_eco: dict, result_short: dict, metric_name: str, 
                               metric_label: str, unit: str, eco_value: float, short_value: float,
                               algorithm_name: str = "") -> str:
    """
    Creates a comparative bar chart for a single metric.
    
    Args:
        result_eco: Dictionary with ecological route data
        result_short: Dictionary with shortest route data
        metric_name: Metric name (for title)
        metric_label: Label for Y axis
        unit: Metric unit
        eco_value: Ecological route value
        short_value: Shortest route value
        algorithm_name: Algorithm name (for title)
    
    Returns:
        Base64 string of PNG image
    """
    # Configuração estilo científico
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Dados
    categories = ['Ecological Route', 'Shortest Route']
    values = [eco_value, short_value]
    colors = ['#1976d2', '#d32f2f']
    
    # Cria barras
    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
    # Formatação
    ax.set_ylabel(f'{metric_label} ({unit})', fontsize=12, fontweight='bold')
    title = f'{metric_name}'
    if algorithm_name:
        title += f' ({algorithm_name})'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            # Formata valores de acordo com a unidade
            if unit == 'm':
                label = f'{height:.0f}'
            elif unit == 'L':
                label = f'{height:.3f}'
            elif unit == 'min':
                label = f'{height:.2f}'
            else:
                label = f'{height:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Converte para base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"


def create_route_comparison_charts(result_eco: dict, result_short: dict, algorithm_name: str = "") -> tuple:
    """
    Creates 3 separate comparative bar charts between Ecological Route and Shortest Route.
    
    Args:
        result_eco: Dictionary with ecological route data
        result_short: Dictionary with shortest route data
        algorithm_name: Algorithm name (for title)
    
    Returns:
        Tuple with 3 base64 strings (distance, fuel, time)
    """
    chart_distance = create_single_metric_chart(
        result_eco, result_short, 'Distance', 'Distance', 'm',
        result_eco.get('total_length_m', 0),
        result_short.get('total_length_m', 0),
        algorithm_name
    )
    
    chart_fuel = create_single_metric_chart(
        result_eco, result_short, 'Fuel Consumption', 'Fuel', 'L',
        result_eco.get('total_fuel_liters', 0),
        result_short.get('total_fuel_liters', 0),
        algorithm_name
    )
    
    chart_time = create_single_metric_chart(
        result_eco, result_short, 'Travel Time', 'Travel Time', 'min',
        result_eco.get('total_time_min', 0),
        result_short.get('total_time_min', 0),
        algorithm_name
    )
    
    return chart_distance, chart_fuel, chart_time


def create_algorithm_comparison_chart(metric_name: str, metric_label: str, unit: str,
                                     dijkstra_eco: float, dijkstra_short: float,
                                     astar_eco: float, astar_short: float) -> str:
    """
    Creates a comparative bar chart between Dijkstra and A* algorithms.
    
    Args:
        metric_name: Metric name (for title)
        metric_label: Label for Y axis
        unit: Metric unit
        dijkstra_eco: Dijkstra ecological route value
        dijkstra_short: Dijkstra shortest route value
        astar_eco: A* ecological route value
        astar_short: A* shortest route value
    
    Returns:
        Base64 string of PNG image
    """
    # Configuração estilo científico
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dados
    categories = ['Ecological Route', 'Shortest Route']
    dijkstra_values = [dijkstra_eco, dijkstra_short]
    astar_values = [astar_eco, astar_short]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Cria barras
    bars1 = ax.bar(x - width/2, dijkstra_values, width,
                   label='Dijkstra', color='#1976d2', alpha=0.8)
    bars2 = ax.bar(x + width/2, astar_values, width,
                   label='A*', color='#7b1fa2', alpha=0.8)
    
    # Formatação
    ax.set_ylabel(f'{metric_label} ({unit})', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison: Dijkstra vs A*', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # Formata valores de acordo com a unidade
                if unit == 'ms':
                    label = f'{height:.2f}'
                elif unit == 'm':
                    label = f'{height:.0f}'
                elif unit == 'L':
                    label = f'{height:.3f}'
                elif unit == 'min':
                    label = f'{height:.2f}'
                else:
                    label = f'{height:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Converte para base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"


def render_both_routes_to_html(start_addr: str, dest_addr: str, output_html: str = "rotas_comparacao.html", zoom_start: int = 14) -> Path:
    """
    Renderiza ambas as rotas (ecológica e mais curta) em dois mapas na mesma página HTML.
    Inclui título, tabela de comparação completa e endereços nos pins.
    
    Args:
        start_addr: Endereço de origem
        dest_addr: Endereço de destino
        output_html: Nome do arquivo HTML de saída
        zoom_start: Nível de zoom inicial do mapa
    
    Returns:
        Path do arquivo HTML gerado
    """
    # Calcula ambas as rotas
    result_short, result_eco = calculate_route_dijkstra(start_addr, dest_addr)
    
    # Carrega grafo para extrair coordenadas
    G = build_graph_from_csv()
    
    # Extrai coordenadas das rotas
    coords_eco = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_eco['path_nodes']]
    coords_short = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_short['path_nodes']]
    
    if not coords_eco or not coords_short:
        raise ValueError("Uma das rotas está vazia, não há coordenadas para desenhar.")
    
    key_points_eco = get_key_points(G, result_eco['path_nodes'])
    key_points_short = get_key_points(G, result_short['path_nodes'])
    
    # Faz reverse geocoding para os pontos-chave (apenas início e fim)
    addresses_eco = {}
    for node_id, lat, lon in key_points_eco:
        addresses_eco[node_id] = reverse_geocode(lat, lon)
        time.sleep(1)  # Rate limiting para Nominatim
    
    addresses_short = {}
    for node_id, lat, lon in key_points_short:
        addresses_short[node_id] = reverse_geocode(lat, lon)
        time.sleep(1)  # Rate limiting para Nominatim
    
    # Cria os dois mapas
    m_eco = folium.Map(location=[coords_eco[0][0], coords_eco[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    m_short = folium.Map(location=[coords_short[0][0], coords_short[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    
    # ========== MAPA ECOLÓGICO ==========
    # Desenha rota ecológica
    folium.PolyLine(
        coords_eco, 
        color="blue", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Ecological Route - {result_eco['total_length_m']:.0f}m, {result_eco['total_fuel_liters']:.3f}L"
    ).add_to(m_eco)
    
    # Adiciona marcadores com endereços (apenas início e fim)
    for node_id, lat, lon in key_points_eco:
        if node_id == result_eco['start_node']:
            icon_color = "green"
            label = "Start"
        elif node_id == result_eco['end_node']:
            icon_color = "red"
            label = "Destination"
        else:
            continue  # Pula qualquer outro ponto
        
        address = addresses_eco.get(node_id, f"Lat: {lat:.6f}, Lon: {lon:.6f}")
        popup_text = f"<b>{label}</b><br>{address}"
        
        folium.Marker(
            (lat, lon),
            icon=folium.Icon(color=icon_color),
            tooltip=label,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m_eco)
    
    # ========== MAPA MAIS CURTO ==========
    # Desenha rota mais curta
    folium.PolyLine(
        coords_short, 
        color="red", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Shortest Route - {result_short['total_length_m']:.0f}m, {result_short['total_fuel_liters']:.3f}L"
    ).add_to(m_short)
    
    # Adiciona marcadores com endereços (apenas início e fim)
    for node_id, lat, lon in key_points_short:
        if node_id == result_short['start_node']:
            icon_color = "green"
            label = "Start"
        elif node_id == result_short['end_node']:
            icon_color = "red"
            label = "Destination"
        else:
            continue  # Pula qualquer outro ponto
        
        address = addresses_short.get(node_id, f"Lat: {lat:.6f}, Lon: {lon:.6f}")
        popup_text = f"<b>{label}</b><br>{address}"
        
        folium.Marker(
            (lat, lon),
            icon=folium.Icon(color=icon_color),
            tooltip=label,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m_short)
    
    # ========== MAPA COMPARATIVO (AMBAS AS ROTAS JUNTAS) ==========
    # Cria um terceiro mapa com ambas as rotas sobrepostas
    center_lat = (coords_eco[0][0] + coords_short[0][0]) / 2
    center_lon = (coords_eco[0][1] + coords_short[0][1]) / 2
    m_comparison = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB positron")
    
    # Desenha rota ecológica (azul)
    folium.PolyLine(
        coords_eco, 
        color="blue", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Ecological Route - {result_eco['total_length_m']:.0f}m, {result_eco['total_fuel_liters']:.3f}L"
    ).add_to(m_comparison)
    
    # Desenha rota mais curta (vermelho)
    folium.PolyLine(
        coords_short, 
        color="red", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Shortest Route - {result_short['total_length_m']:.0f}m, {result_short['total_fuel_liters']:.3f}L"
    ).add_to(m_comparison)
    
    # Adiciona marcadores de início e fim (apenas uma vez)
    folium.Marker(
        coords_eco[0], 
        icon=folium.Icon(color="green"), 
        tooltip="Start"
    ).add_to(m_comparison)
    folium.Marker(
        coords_eco[-1], 
        icon=folium.Icon(color="red"), 
        tooltip="Destination"
    ).add_to(m_comparison)

    # ========== CRIA HTML COMBINADO ==========
    # Salva os mapas temporariamente para obter o HTML completo
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_eco:
        m_eco.save(tmp_eco.name)
        tmp_eco_path = tmp_eco.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_short:
        m_short.save(tmp_short.name)
        tmp_short_path = tmp_short.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_comparison:
        m_comparison.save(tmp_comparison.name)
        tmp_comparison_path = tmp_comparison.name
    
    # Lê o HTML completo dos arquivos temporários
    with open(tmp_eco_path, 'r', encoding='utf-8') as f:
        eco_html = f.read()
    
    with open(tmp_short_path, 'r', encoding='utf-8') as f:
        short_html = f.read()
    
    with open(tmp_comparison_path, 'r', encoding='utf-8') as f:
        comparison_html = f.read()
    
    # Remove arquivos temporários
    try:
        os.unlink(tmp_eco_path)
        os.unlink(tmp_short_path)
        os.unlink(tmp_comparison_path)
    except:
        pass
    
    # Extrai o conteúdo necessário de cada HTML
    # Mapa ecológico
    eco_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', eco_html, re.DOTALL)
    eco_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', eco_html)
    eco_map_id = eco_map_id_match.group(1) if eco_map_id_match else "map_eco_temp"
    
    # Mapa mais curto
    short_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', short_html, re.DOTALL)
    short_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', short_html)
    short_map_id = short_map_id_match.group(1) if short_map_id_match else "map_short_temp"
    
    # Mapa comparativo
    comparison_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', comparison_html, re.DOTALL)
    comparison_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', comparison_html)
    comparison_map_id = comparison_map_id_match.group(1) if comparison_map_id_match else "map_comparison_temp"
    
    # Extrai todos os scripts de todos os mapas
    eco_scripts = re.findall(r'<script[^>]*>.*?</script>', eco_html, re.DOTALL)
    short_scripts = re.findall(r'<script[^>]*>.*?</script>', short_html, re.DOTALL)
    comparison_scripts = re.findall(r'<script[^>]*>.*?</script>', comparison_html, re.DOTALL)
    
    # Extrai head (CSS, etc) - pega apenas uma vez
    head_match = re.search(r'<head>(.*?)</head>', eco_html, re.DOTALL)
    head_content = head_match.group(1) if head_match else ""
    
    # Cria HTML combinado
    # Calcula comparação localmente
    comp_dijkstra = {
        'length_diff_m': result_eco['total_length_m'] - result_short['total_length_m'],
        'fuel_diff_liters': result_short['total_fuel_liters'] - result_eco['total_fuel_liters'],
        'time_diff_min': result_eco['total_time_min'] - result_short['total_time_min'],
    }

    fuel_diff_dijkstra = comp_dijkstra['fuel_diff_liters']
    length_diff_dijkstra = comp_dijkstra['length_diff_m']
    time_diff_dijkstra = abs(comp_dijkstra['time_diff_min'])
    
    # Gera gráficos comparativos de rotas (3 gráficos separados)
    chart_dijkstra_distance, chart_dijkstra_fuel, chart_dijkstra_time = create_route_comparison_charts(result_eco, result_short, "Dijkstra")
    
    # Prepara o conteúdo dos mapas
    eco_map_content = eco_map_match.group(1) if eco_map_match else ""
    short_map_content = short_map_match.group(1) if short_map_match else ""
    comparison_map_content = comparison_map_match.group(1) if comparison_map_match else ""
    
    # Substitui IDs nos scripts para evitar conflitos
    eco_scripts_clean = []
    for script in eco_scripts:
        script_clean = script.replace(eco_map_id, 'map_eco_leaflet')
        eco_scripts_clean.append(script_clean)
    
    short_scripts_clean = []
    for script in short_scripts:
        script_clean = script.replace(short_map_id, 'map_short_leaflet')
        short_scripts_clean.append(script_clean)
    
    comparison_scripts_clean = []
    for script in comparison_scripts:
        script_clean = script.replace(comparison_map_id, 'map_comparison_leaflet')
        comparison_scripts_clean.append(script_clean)
    
    combined_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <title>Ecological Route vs Shortest Route</title>
    {head_content}
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-radius: 0 0 20px 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        .main-content {{
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .maps-section {{
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .map-card {{
            flex: 1;
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .map-card-header {{
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 2px solid #e0e0e0;
        }}
        .map-card-header h2 {{
            margin: 0;
            font-size: 22px;
            font-weight: 600;
            color: #333;
        }}
        .map-card:first-child .map-card-header {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-bottom-color: #1976d2;
        }}
        .map-card:first-child .map-card-header h2 {{
            color: #1976d2;
        }}
        .map-card:last-child .map-card-header {{
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-bottom-color: #d32f2f;
        }}
        .map-card:last-child .map-card-header h2 {{
            color: #d32f2f;
        }}
        .map-container {{
            position: relative;
            height: 500px;
            width: 100%;
        }}
        .map-container .map {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        #map_eco_leaflet, #map_short_leaflet, #map_comparison_leaflet {{
            width: 100%;
            height: 100%;
            border-radius: 0;
        }}
        .map-info {{
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }}
        .map-info h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }}
        .map-info p {{
            margin: 8px 0;
            font-size: 14px;
            line-height: 1.6;
            color: #555;
        }}
        .map-info .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .map-info .metric:last-child {{
            border-bottom: none;
        }}
        .map-info .metric-label {{
            font-weight: 500;
            color: #666;
        }}
        .map-info .metric-value {{
            font-weight: 600;
            color: #333;
        }}
        .comparison-map-section {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }}
        .comparison-map-section h2 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            font-weight: 600;
            color: #333;
            text-align: center;
        }}
        .comparison-map-container {{
            position: relative;
            height: 600px;
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #e0e0e0;
        }}
        .comparison-map-container .map {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        .comparison-legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .comparison-legend h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        .comparison-legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 14px;
        }}
        .comparison-legend-color {{
            width: 20px;
            height: 4px;
            margin-right: 10px;
            border-radius: 2px;
        }}
        .comparison-legend-color.eco {{
            background-color: #1976d2;
        }}
        .comparison-legend-color.short {{
            background-color: #d32f2f;
        }}
        .analysis-section {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .analysis-section h2 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            font-weight: 600;
            color: #333;
            text-align: center;
        }}
        .analysis-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }}
        .analysis-box {{
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
        }}
        .analysis-box.eco {{
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-color: #4caf50;
        }}
        .analysis-box.short {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-color: #ff9800;
        }}
        .analysis-box h3 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }}
        .analysis-box ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .analysis-box li {{
            margin: 10px 0;
            font-size: 14px;
            line-height: 1.6;
            color: #555;
        }}
        .analysis-box .advantage {{
            color: #2e7d32;
            font-weight: 500;
        }}
        .analysis-box .disadvantage {{
            color: #c62828;
            font-weight: 500;
        }}
        .chart-section {{
            margin-top: 30px;
            padding: 25px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .chart-section h3 {{
            margin: 0 0 20px 0;
            font-size: 20px;
            font-weight: 600;
            color: #333;
            text-align: center;
        }}
        .chart-section img {{
            width: 100%;
            max-width: 1000px;
            display: block;
            margin: 0 auto;
            border-radius: 8px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }}
        .chart-item {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-item h4 {{
            margin: 0 0 15px 0;
            font-size: 16px;
            font-weight: 600;
            color: #333;
            text-align: center;
        }}
        @media (max-width: 1200px) {{
            .maps-section {{
                flex-direction: column;
            }}
            .map-container {{
                height: 400px;
            }}
            .comparison-map-container {{
                height: 500px;
            }}
            .analysis-content {{
                grid-template-columns: 1fr;
            }}
            .charts-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        @media (max-width: 768px) {{
            .main-content {{
                padding: 10px;
            }}
            .comparison-map-container {{
                height: 400px;
            }}
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DIJKSTRA</h1>
    </div>
    
    <div class="main-content">
        <div class="maps-section">
            <div class="map-card">
                <div class="map-card-header">
                    <h2>Ecological Route</h2>
                </div>
                <div class="map-container">
                    <div class="map" id="map_eco">
                        <div id="map_eco_leaflet">
                            {eco_map_content}
                        </div>
                    </div>
                </div>
                <div class="map-info">
                    <h3>Route Information</h3>
                    <div class="metric">
                        <span class="metric-label">Total Distance:</span>
                        <span class="metric-value">{result_eco['total_length_m']:.1f} meters</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Estimated Time:</span>
                        <span class="metric-value">{result_eco['total_time_min']:.1f} minutes</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Fuel Consumption:</span>
                        <span class="metric-value">{result_eco['total_fuel_liters']:.3f} liters</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #666;">
                        This route was optimized to minimize fuel consumption, considering factors such as terrain slope and road speed limits.
                    </p>
                </div>
            </div>
            
            <div class="map-card">
                <div class="map-card-header">
                    <h2>Shortest Route</h2>
                </div>
                <div class="map-container">
                    <div class="map" id="map_short">
                        <div id="map_short_leaflet">
                            {short_map_content}
                        </div>
                    </div>
                </div>
                <div class="map-info">
                    <h3>Route Information</h3>
                    <div class="metric">
                        <span class="metric-label">Total Distance:</span>
                        <span class="metric-value">{result_short['total_length_m']:.1f} meters</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Estimated Time:</span>
                        <span class="metric-value">{result_short['total_time_min']:.1f} minutes</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Fuel Consumption:</span>
                        <span class="metric-value">{result_short['total_fuel_liters']:.3f} liters</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #666;">
                        This route prioritizes the shortest possible distance between origin and destination points.
                    </p>
                </div>
            </div>
        </div>
        
        <div class="comparison-map-section">
            <h2>Visual Route Comparison</h2>
            <div class="comparison-map-container">
                <div class="map" id="map_comparison">
                    <div id="map_comparison_leaflet">
                        {comparison_map_content}
                    </div>
                </div>
                <div class="comparison-legend">
                    <h3>Legend</h3>
                    <div class="comparison-legend-item">
                        <div class="comparison-legend-color eco"></div>
                        <span>Ecological Route</span>
                    </div>
                    <div class="comparison-legend-item">
                        <div class="comparison-legend-color short"></div>
                        <span>Shortest Route</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>Comparative Analysis: Advantages and Disadvantages</h2>
            <div class="analysis-content">
                <div class="analysis-box eco">
                    <h3>Ecological Route</h3>
                    <ul>
                        <li><span class="advantage">✓ Advantage:</span> Lower fuel consumption ({fuel_diff_dijkstra:+.3f}L difference)</li>
                        <li><span class="advantage">✓ Advantage:</span> {'Shorter travel time' if comp_dijkstra['time_diff_min'] < 0 else 'Similar time'}</li>
                        <li><span class="advantage">✓ Advantage:</span> More sustainable and economical in the long term</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> {'Slightly longer distance' if comp_dijkstra['length_diff_m'] > 0 else 'Similar distance'} ({length_diff_dijkstra:+.1f}m)</li>
                    </ul>
                </div>
                
                <div class="analysis-box short">
                    <h3>Shortest Route</h3>
                    <ul>
                        <li><span class="advantage">✓ Advantage:</span> Shorter distance traveled ({length_diff_dijkstra:+.1f}m difference)</li>
                        <li><span class="advantage">✓ Advantage:</span> More direct path between origin and destination</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> Higher fuel consumption ({fuel_diff_dijkstra:+.3f}L more)</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> {'Longer travel time' if comp_dijkstra['time_diff_min'] > 0 else 'Similar time'} ({time_diff_dijkstra:+.1f} min)</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="chart-section">
            <h3>Visual Comparison: Ecological Route vs Shortest Route (Dijkstra)</h3>
            <div class="charts-grid">
                <div class="chart-item">
                    <h4>Distance</h4>
                    <img src="{chart_dijkstra_distance}" alt="Dijkstra Distance Chart" style="width: 100%; border-radius: 8px;">
                </div>
                <div class="chart-item">
                    <h4>Fuel Consumption</h4>
                    <img src="{chart_dijkstra_fuel}" alt="Dijkstra Fuel Chart" style="width: 100%; border-radius: 8px;">
                </div>
                <div class="chart-item">
                    <h4>Travel Time</h4>
                    <img src="{chart_dijkstra_time}" alt="Dijkstra Time Chart" style="width: 100%; border-radius: 8px;">
                </div>
            </div>
        </div>
    </div>
    
    {''.join(eco_scripts_clean)}
    {''.join(short_scripts_clean)}
    {''.join(comparison_scripts_clean)}
</body>
</html>'''
    
    # Salva o HTML combinado
    out = Path(output_html).resolve()
    with open(out, 'w', encoding='utf-8') as f:
        f.write(combined_html)
    
    return out


def render_all_routes_combined(start_addr: str, dest_addr: str, output_html: str = "rotas_completo.html", zoom_start: int = 14) -> Path:
    """
    Renderiza todas as rotas (Dijkstra e A*) em um único arquivo HTML.
    Reutiliza render_both_routes_to_html e adiciona seção A*.
    """
    # Gera HTML do Dijkstra primeiro
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
        render_both_routes_to_html(start_addr, dest_addr, tmp.name, zoom_start)
        dijkstra_path = tmp.name
    
    with open(dijkstra_path, 'r', encoding='utf-8') as f:
        dijkstra_html = f.read()
    
    try:
        os.unlink(dijkstra_path)
    except:
        pass
    
    # Calcula rotas A* e mede tempo total
    astar_start_time = time.perf_counter()
    result_eco_astar, result_short_astar = calculate_astar_routes(start_addr, dest_addr)
    astar_total_time = time.perf_counter() - astar_start_time
    
    # Calcula também Dijkstra para comparação
    dijkstra_start_time = time.perf_counter()
    result_short_dijkstra, result_eco_dijkstra = calculate_route_dijkstra(start_addr, dest_addr)
    dijkstra_total_time = time.perf_counter() - dijkstra_start_time
    
    # Carrega grafo para extrair coordenadas
    G = build_graph_from_csv()
    
    # Cria mapas A* (similar ao que fazemos em render_both_routes_to_html)
    coords_eco_astar = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_eco_astar['path_nodes']]
    coords_short_astar = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_short_astar['path_nodes']]
    
    m_eco_astar = folium.Map(location=[coords_eco_astar[0][0], coords_eco_astar[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    m_short_astar = folium.Map(location=[coords_short_astar[0][0], coords_short_astar[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    
    key_points_astar = get_key_points(G, result_eco_astar['path_nodes'])
    addresses_astar = {}
    for node_id, lat, lon in key_points_astar:
        addresses_astar[node_id] = reverse_geocode(lat, lon)
        time.sleep(1)
    
    folium.PolyLine(coords_eco_astar, color="blue", weight=5, opacity=0.8, tooltip=f"Ecological Route A* - {result_eco_astar['total_length_m']:.0f}m").add_to(m_eco_astar)
    folium.PolyLine(coords_short_astar, color="red", weight=5, opacity=0.8, tooltip=f"Shortest Route A* - {result_short_astar['total_length_m']:.0f}m").add_to(m_short_astar)
    
    for node_id, lat, lon in key_points_astar:
        if node_id == result_eco_astar['start_node']:
            address = addresses_astar.get(node_id, f"Lat: {lat:.6f}, Lon: {lon:.6f}")
            folium.Marker((lat, lon), icon=folium.Icon(color="green"), tooltip="Start", popup=folium.Popup(f"<b>Start</b><br>{address}", max_width=300)).add_to(m_eco_astar)
            folium.Marker((lat, lon), icon=folium.Icon(color="green"), tooltip="Start", popup=folium.Popup(f"<b>Start</b><br>{address}", max_width=300)).add_to(m_short_astar)
        elif node_id == result_eco_astar['end_node']:
            address = addresses_astar.get(node_id, f"Lat: {lat:.6f}, Lon: {lon:.6f}")
            folium.Marker((lat, lon), icon=folium.Icon(color="red"), tooltip="Destination", popup=folium.Popup(f"<b>Destination</b><br>{address}", max_width=300)).add_to(m_eco_astar)
            folium.Marker((lat, lon), icon=folium.Icon(color="red"), tooltip="Destination", popup=folium.Popup(f"<b>Destination</b><br>{address}", max_width=300)).add_to(m_short_astar)
    
    # ========== MAPA COMPARATIVO A* (ambas as rotas sobrepostas) ==========
    center_lat_astar = (coords_eco_astar[0][0] + coords_short_astar[0][0]) / 2
    center_lon_astar = (coords_eco_astar[0][1] + coords_short_astar[0][1]) / 2
    m_comparison_astar = folium.Map(location=[center_lat_astar, center_lon_astar], zoom_start=zoom_start, tiles="CartoDB positron")
    
    folium.PolyLine(coords_eco_astar, color="blue", weight=5, opacity=0.8, tooltip="Ecological Route A*").add_to(m_comparison_astar)
    folium.PolyLine(coords_short_astar, color="red", weight=5, opacity=0.8, tooltip="Shortest Route A*").add_to(m_comparison_astar)
    folium.Marker(coords_eco_astar[0], icon=folium.Icon(color="green"), tooltip="Start").add_to(m_comparison_astar)
    folium.Marker(coords_eco_astar[-1], icon=folium.Icon(color="red"), tooltip="Destination").add_to(m_comparison_astar)
    
    # Salva mapas A* temporariamente
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_eco:
        m_eco_astar.save(tmp_eco.name)
        tmp_eco_path = tmp_eco.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_short:
        m_short_astar.save(tmp_short.name)
        tmp_short_path = tmp_short.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_comparison_astar:
        m_comparison_astar.save(tmp_comparison_astar.name)
        tmp_comparison_astar_path = tmp_comparison_astar.name
    
    with open(tmp_eco_path, 'r', encoding='utf-8') as f:
        eco_astar_html = f.read()
    with open(tmp_short_path, 'r', encoding='utf-8') as f:
        short_astar_html = f.read()
    with open(tmp_comparison_astar_path, 'r', encoding='utf-8') as f:
        comparison_astar_html = f.read()
    
    try:
        os.unlink(tmp_eco_path)
        os.unlink(tmp_short_path)
        os.unlink(tmp_comparison_astar_path)
    except:
        pass
    
    # Extrai conteúdo dos mapas A*
    eco_astar_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', eco_astar_html, re.DOTALL)
    eco_astar_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', eco_astar_html)
    eco_astar_map_id = eco_astar_map_id_match.group(1) if eco_astar_map_id_match else "map_eco_astar_temp"
    
    short_astar_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', short_astar_html, re.DOTALL)
    short_astar_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', short_astar_html)
    short_astar_map_id = short_astar_map_id_match.group(1) if short_astar_map_id_match else "map_short_astar_temp"
    
    comparison_astar_map_match = re.search(r'<div[^>]*id="map[^"]*"[^>]*>(.*?)</div>\s*</body>', comparison_astar_html, re.DOTALL)
    comparison_astar_map_id_match = re.search(r'<div[^>]*id="(map[^"]*)"', comparison_astar_html)
    comparison_astar_map_id = comparison_astar_map_id_match.group(1) if comparison_astar_map_id_match else "map_comparison_astar_temp"
    
    eco_astar_scripts = re.findall(r'<script[^>]*>.*?</script>', eco_astar_html, re.DOTALL)
    short_astar_scripts = re.findall(r'<script[^>]*>.*?</script>', short_astar_html, re.DOTALL)
    comparison_astar_scripts = re.findall(r'<script[^>]*>.*?</script>', comparison_astar_html, re.DOTALL)
    
    eco_astar_map_content = eco_astar_map_match.group(1) if eco_astar_map_match else ""
    short_astar_map_content = short_astar_map_match.group(1) if short_astar_map_match else ""
    comparison_astar_map_content = comparison_astar_map_match.group(1) if comparison_astar_map_match else ""
    
    eco_astar_scripts_clean = [s.replace(eco_astar_map_id, 'map_eco_astar_leaflet') for s in eco_astar_scripts]
    short_astar_scripts_clean = [s.replace(short_astar_map_id, 'map_short_astar_leaflet') for s in short_astar_scripts]
    comparison_astar_scripts_clean = [s.replace(comparison_astar_map_id, 'map_comparison_astar_leaflet') for s in comparison_astar_scripts]
    
    # Comparison between A* routes (ecological vs shortest) - same structure as Dijkstra
    comp_astar = {
        'length_diff_m': result_eco_astar['total_length_m'] - result_short_astar['total_length_m'],
        'fuel_diff_liters': result_short_astar['total_fuel_liters'] - result_eco_astar['total_fuel_liters'],
        'time_diff_min': result_eco_astar['total_time_min'] - result_short_astar['total_time_min'],
    }
    
    # Ajusta os textos para mostrar valores absolutos quando necessário
    fuel_diff_astar = comp_astar['fuel_diff_liters']
    length_diff_astar = comp_astar['length_diff_m']
    time_diff_astar = abs(comp_astar['time_diff_min'])
    
    # Gera gráficos comparativos de rotas do A* (3 gráficos separados)
    chart_astar_distance, chart_astar_fuel, chart_astar_time = create_route_comparison_charts(result_eco_astar, result_short_astar, "A*")
    
    # Gera gráfico comparativo de performance entre algoritmos (apenas tempo de execução)
    chart_execution_time = create_algorithm_comparison_chart(
        "Execution Time", "Time", "ms",
        result_eco_dijkstra.get('execution_time_seconds', 0) * 1000,
        result_short_dijkstra.get('execution_time_seconds', 0) * 1000,
        result_eco_astar.get('execution_time_seconds', 0) * 1000,
        result_short_astar.get('execution_time_seconds', 0) * 1000
    )
    
    # Adiciona seção A* ao HTML do Dijkstra
    astar_section = f'''
        <!-- SEÇÃO A* -->
        <div class="algorithm-section" style="margin-top: 50px; padding-top: 30px; border-top: 3px solid #1976d2;">
            <div class="main-content">
                <h2 style="font-size: 28px; font-weight: 600; color: #333; text-align: center; margin-bottom: 30px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">A* Algorithm</h2>
                
                <div class="maps-section">
                <div class="map-card">
                    <div class="map-card-header">
                        <h2>Ecological Route A*</h2>
                    </div>
                    <div class="map-container">
                        <div class="map" id="map_eco_astar">
                            <div id="map_eco_astar_leaflet">
                                {eco_astar_map_content}
                            </div>
                        </div>
                    </div>
                    <div class="map-info">
                        <h3>Route Information</h3>
                        <div class="metric">
                            <span class="metric-label">Total Distance:</span>
                            <span class="metric-value">{result_eco_astar['total_length_m']:.1f} meters</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Estimated Time:</span>
                            <span class="metric-value">{result_eco_astar['total_time_min']:.1f} minutes</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Fuel Consumption:</span>
                            <span class="metric-value">{result_eco_astar['total_fuel_liters']:.3f} liters</span>
                        </div>
                    </div>
                </div>
                
                <div class="map-card">
                    <div class="map-card-header">
                        <h2>Shortest Route A*</h2>
                    </div>
                    <div class="map-container">
                        <div class="map" id="map_short_astar">
                            <div id="map_short_astar_leaflet">
                                {short_astar_map_content}
                            </div>
                        </div>
                    </div>
                    <div class="map-info">
                        <h3>Route Information</h3>
                        <div class="metric">
                            <span class="metric-label">Total Distance:</span>
                            <span class="metric-value">{result_short_astar['total_length_m']:.1f} meters</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Estimated Time:</span>
                            <span class="metric-value">{result_short_astar['total_time_min']:.1f} minutes</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Fuel Consumption:</span>
                            <span class="metric-value">{result_short_astar['total_fuel_liters']:.3f} liters</span>
                        </div>
                    </div>
                </div>
                </div>
                
                <div class="comparison-map-section">
                    <h2>Visual Route Comparison (A*)</h2>
                    <div class="comparison-map-container">
                        <div class="map" id="map_comparison_astar">
                            <div id="map_comparison_astar_leaflet">
                                {comparison_astar_map_content}
                            </div>
                        </div>
                        <div class="comparison-legend">
                            <h3>Legend</h3>
                            <div class="comparison-legend-item">
                                <div class="comparison-legend-color eco"></div>
                                <span>Ecological Route</span>
                            </div>
                            <div class="comparison-legend-item">
                                <div class="comparison-legend-color short"></div>
                                <span>Shortest Route</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- ANÁLISE COMPARATIVA DAS ROTAS A* -->
                <div class="analysis-section">
                    <h2>Comparative Analysis: Advantages and Disadvantages (A*)</h2>
                    <div class="analysis-content">
                        <div class="analysis-box eco">
                    <h3>Ecological Route A*</h3>
                    <ul>
                        <li><span class="advantage">✓ Advantage:</span> Lower fuel consumption ({fuel_diff_astar:+.3f}L difference)</li>
                        <li><span class="advantage">✓ Advantage:</span> {'Shorter travel time' if comp_astar['time_diff_min'] < 0 else 'Similar time'}</li>
                        <li><span class="advantage">✓ Advantage:</span> More sustainable and economical in the long term</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> {'Slightly longer distance' if comp_astar['length_diff_m'] > 0 else 'Similar distance'} ({length_diff_astar:+.1f}m)</li>
                    </ul>
                </div>
                
                <div class="analysis-box short">
                    <h3>Shortest Route A*</h3>
                    <ul>
                        <li><span class="advantage">✓ Advantage:</span> Shorter distance traveled ({length_diff_astar:+.1f}m difference)</li>
                        <li><span class="advantage">✓ Advantage:</span> More direct path between origin and destination</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> Higher fuel consumption ({fuel_diff_astar:+.3f}L more)</li>
                        <li><span class="disadvantage">✗ Disadvantage:</span> {'Longer travel time' if comp_astar['time_diff_min'] < 0 else 'Similar time'} ({time_diff_astar:+.1f} min)</li>
                    </ul>
                        </div>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h3>Visual Comparison: Ecological Route vs Shortest Route (A*)</h3>
                    <div class="charts-grid">
                        <div class="chart-item">
                            <h4>Distance</h4>
                            <img src="{chart_astar_distance}" alt="A* Distance Chart" style="width: 100%; border-radius: 8px;">
                        </div>
                        <div class="chart-item">
                            <h4>Fuel Consumption</h4>
                            <img src="{chart_astar_fuel}" alt="A* Fuel Chart" style="width: 100%; border-radius: 8px;">
                        </div>
                        <div class="chart-item">
                            <h4>Travel Time</h4>
                            <img src="{chart_astar_time}" alt="A* Time Chart" style="width: 100%; border-radius: 8px;">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- COMPARAÇÃO DE PERFORMANCE: DIJKSTRA vs A* -->
        <div class="performance-section" style="margin-top: 50px; padding-top: 30px; border-top: 3px solid #9c27b0;">
            <div class="main-content">
                <h2 style="font-size: 28px; font-weight: 600; color: #333; text-align: center; margin-bottom: 30px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Algorithm Performance Comparison</h2>
            
            <div class="performance-content" style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 20px;">
                <div class="performance-box" style="padding: 25px; border-radius: 12px; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border: 2px solid #2196f3; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 600; color: #1976d2;">Dijkstra Algorithm</h3>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Total Execution Time:</span>
                        <span style="font-weight: 700; color: #1976d2; font-size: 18px; margin-left: 10px;">{dijkstra_total_time*1000:.2f} ms</span>
                    </div>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Ecological Route:</span>
                        <span style="font-weight: 600; color: #333; margin-left: 10px;">{result_eco_dijkstra.get('execution_time_seconds', 0)*1000:.2f} ms</span>
                    </div>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Shortest Route:</span>
                        <span style="font-weight: 600; color: #333; margin-left: 10px;">{result_short_dijkstra.get('execution_time_seconds', 0)*1000:.2f} ms</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #555; font-size: 14px;">
                        Classic graph search algorithm that explores all possible paths until finding the optimal solution.
                    </p>
                </div>
                
                <div class="performance-box" style="padding: 25px; border-radius: 12px; background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border: 2px solid #9c27b0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 600; color: #7b1fa2;">A* Algorithm</h3>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Total Execution Time:</span>
                        <span style="font-weight: 700; color: #7b1fa2; font-size: 18px; margin-left: 10px;">{astar_total_time*1000:.2f} ms</span>
                    </div>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Ecological Route:</span>
                        <span style="font-weight: 600; color: #333; margin-left: 10px;">{result_eco_astar.get('execution_time_seconds', 0)*1000:.2f} ms</span>
                    </div>
                    <div class="performance-metric" style="margin-bottom: 12px;">
                        <span style="font-weight: 500; color: #666;">Shortest Route:</span>
                        <span style="font-weight: 600; color: #333; margin-left: 10px;">{result_short_astar.get('execution_time_seconds', 0)*1000:.2f} ms</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #555; font-size: 14px;">
                        Heuristic algorithm that uses information about the destination to optimize the search.
                    </p>
                </div>
            </div>
            
            <div class="performance-comparison" style="margin-top: 30px; padding: 25px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 20px 0; font-size: 20px; font-weight: 600; color: #333; text-align: center;">Performance Analysis</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600; color: #1976d2;">Time Difference:</h4>
                        <p style="margin: 0; color: #555; font-size: 15px;">
                            {'A* was' if astar_total_time < dijkstra_total_time else 'Dijkstra was'} 
                            <strong style="color: #1976d2;">{abs((astar_total_time - dijkstra_total_time) / max(dijkstra_total_time, 0.0001) * 100):.1f}%</strong>
                            {'faster' if astar_total_time < dijkstra_total_time else 'faster'}
                            ({abs(astar_total_time - dijkstra_total_time)*1000:.2f} ms difference)
                        </p>
                    </div>
                    <div>
                        <h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600; color: #7b1fa2;">Efficiency:</h4>
                        <p style="margin: 0; color: #555; font-size: 15px;">
                            {'A* uses heuristics to reduce the search space, resulting in' if astar_total_time < dijkstra_total_time else 'Dijkstra systematically explores all paths, ensuring'}
                            {'shorter execution times.' if astar_total_time < dijkstra_total_time else 'optimality with possible higher computational cost.'}
                        </p>
                    </div>
                    </div>
                </div>
                
                <div class="chart-section" style="margin-top: 30px;">
                    <h3 style="margin: 0 0 20px 0; font-size: 20px; font-weight: 600; color: #333; text-align: center;">
                        Performance Visual Comparison: Execution Time - Dijkstra vs A*
                    </h3>
                    <img src="{chart_execution_time}" alt="Execution Time Chart" style="width: 100%; max-width: 1000px; display: block; margin: 0 auto; border-radius: 8px;">
                </div>
            </div>
            </div>
        </div>
        
        {''.join(eco_astar_scripts_clean)}
        {''.join(short_astar_scripts_clean)}
        {''.join(comparison_astar_scripts_clean)}
    '''
    
    # Insere seção A* antes do </body> e atualiza título
    combined_html = dijkstra_html.replace('</body>', astar_section + '</body>')
    combined_html = combined_html.replace(
        '<h1>A*</h1>',
        '<h1>DIJKSTRA</h1>'
    )
    
    # Adiciona CSS para IDs A*
    combined_html = combined_html.replace(
        '#map_eco_leaflet, #map_short_leaflet, #map_comparison_leaflet {',
        '#map_eco_leaflet, #map_short_leaflet, #map_comparison_leaflet, #map_eco_astar_leaflet, #map_short_astar_leaflet, #map_comparison_astar_leaflet {'
    )
    
    out = Path(output_html).resolve()
    with open(out, 'w', encoding='utf-8') as f:
        f.write(combined_html)
    
    return out

