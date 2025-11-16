# /Users/hugohenriquemarques/Desktop/Trabalho Final AED2/Repositorio/TrabFinalAEDS2/calculation/render_map.py
from pathlib import Path
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import re
from typing import Tuple
from .routing import build_graph_from_csv, route_ecological, route_shortest_distance, compare_routes

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


def get_key_points(G, path_nodes, max_points: int = 10):
    """
    Seleciona apenas os pontos de início e fim da rota.
    
    Args:
        G: Grafo
        path_nodes: Lista de nós do caminho
        max_points: Ignorado, sempre retorna apenas início e fim
    
    Returns:
        Lista de tuplas (node_id, lat, lon) com apenas início e fim
    """
    # Retorna apenas início e fim
    if len(path_nodes) < 2:
        return [(path_nodes[0], float(G.nodes[path_nodes[0]]['y']), float(G.nodes[path_nodes[0]]['x']))]
    
    return [
        (path_nodes[0], float(G.nodes[path_nodes[0]]['y']), float(G.nodes[path_nodes[0]]['x'])),
        (path_nodes[-1], float(G.nodes[path_nodes[-1]]['y']), float(G.nodes[path_nodes[-1]]['x']))
    ]


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
    G = build_graph_from_csv()
    
    # Calcula ambas as rotas e comparação
    print("Calculando rotas e comparação...")
    comparison = compare_routes(G, start_addr, dest_addr)
    
    result_eco = comparison['ecological']
    result_short = comparison['shortest']
    
    # Extrai coordenadas das rotas
    coords_eco = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_eco['path_nodes']]
    coords_short = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_short['path_nodes']]
    
    if not coords_eco or not coords_short:
        raise ValueError("Uma das rotas está vazia, não há coordenadas para desenhar.")
    
    # Obtém pontos-chave para reverse geocoding (apenas início e fim)
    print("Obtendo endereços dos pontos principais...")
    key_points_eco = get_key_points(G, result_eco['path_nodes'])
    key_points_short = get_key_points(G, result_short['path_nodes'])
    
    # Faz reverse geocoding para os pontos-chave (apenas início e fim)
    addresses_eco = {}
    for node_id, lat, lon in key_points_eco:
        print(f"  Geocodificando ponto ecológico {node_id}...")
        addresses_eco[node_id] = reverse_geocode(lat, lon)
        time.sleep(1)  # Rate limiting para Nominatim
    
    addresses_short = {}
    for node_id, lat, lon in key_points_short:
        print(f"  Geocodificando ponto mais curto {node_id}...")
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
        tooltip=f"Rota Ecológica - {result_eco['total_length_m']:.0f}m, {result_eco['total_fuel_liters']:.3f}L"
    ).add_to(m_eco)
    
    # Adiciona marcadores com endereços (apenas início e fim)
    for node_id, lat, lon in key_points_eco:
        if node_id == result_eco['start_node']:
            icon_color = "green"
            label = "Início"
        elif node_id == result_eco['end_node']:
            icon_color = "red"
            label = "Destino"
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
        tooltip=f"Rota Mais Curta - {result_short['total_length_m']:.0f}m, {result_short['total_fuel_liters']:.3f}L"
    ).add_to(m_short)
    
    # Adiciona marcadores com endereços (apenas início e fim)
    for node_id, lat, lon in key_points_short:
        if node_id == result_short['start_node']:
            icon_color = "green"
            label = "Início"
        elif node_id == result_short['end_node']:
            icon_color = "red"
            label = "Destino"
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
        tooltip=f"Rota Ecológica - {result_eco['total_length_m']:.0f}m, {result_eco['total_fuel_liters']:.3f}L"
    ).add_to(m_comparison)
    
    # Desenha rota mais curta (vermelho)
    folium.PolyLine(
        coords_short, 
        color="red", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Rota Mais Curta - {result_short['total_length_m']:.0f}m, {result_short['total_fuel_liters']:.3f}L"
    ).add_to(m_comparison)
    
    # Adiciona marcadores de início e fim (apenas uma vez)
    folium.Marker(
        coords_eco[0], 
        icon=folium.Icon(color="green"), 
        tooltip="Início"
    ).add_to(m_comparison)
    folium.Marker(
        coords_eco[-1], 
        icon=folium.Icon(color="red"), 
        tooltip="Destino"
    ).add_to(m_comparison)

    # ========== CRIA HTML COMBINADO ==========
    # Salva os mapas temporariamente para obter o HTML completo
    import tempfile
    import os
    
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
    comp = comparison['comparison']
    
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
        }}
        @media (max-width: 768px) {{
            .main-content {{
                padding: 10px;
            }}
            .comparison-map-container {{
                height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ROTAS A PARTIR DO DIJKSTRA</h1>
    </div>
    
    <div class="main-content">
        <div class="maps-section">
            <div class="map-card">
                <div class="map-card-header">
                    <h2>Rota Ecológica</h2>
                </div>
                <div class="map-container">
                    <div class="map" id="map_eco">
                        <div id="map_eco_leaflet">
                            {eco_map_content}
                        </div>
                    </div>
                </div>
                <div class="map-info">
                    <h3>Informações da Rota</h3>
                    <div class="metric">
                        <span class="metric-label">Distância Total:</span>
                        <span class="metric-value">{result_eco['total_length_m']:.1f} metros</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Tempo Estimado:</span>
                        <span class="metric-value">{result_eco['total_time_min']:.1f} minutos</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Consumo de Combustível:</span>
                        <span class="metric-value">{result_eco['total_fuel_liters']:.3f} litros</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #666;">
                        Esta rota foi otimizada para minimizar o consumo de combustível, considerando fatores como inclinação do terreno e velocidade das vias.
                    </p>
                </div>
            </div>
            
            <div class="map-card">
                <div class="map-card-header">
                    <h2>Rota Mais Curta</h2>
                </div>
                <div class="map-container">
                    <div class="map" id="map_short">
                        <div id="map_short_leaflet">
                            {short_map_content}
                        </div>
                    </div>
                </div>
                <div class="map-info">
                    <h3>Informações da Rota</h3>
                    <div class="metric">
                        <span class="metric-label">Distância Total:</span>
                        <span class="metric-value">{result_short['total_length_m']:.1f} metros</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Tempo Estimado:</span>
                        <span class="metric-value">{result_short['total_time_min']:.1f} minutos</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Consumo de Combustível:</span>
                        <span class="metric-value">{result_short['total_fuel_liters']:.3f} litros</span>
                    </div>
                    <p style="margin-top: 15px; font-style: italic; color: #666;">
                        Esta rota prioriza a menor distância possível entre os pontos de origem e destino.
                    </p>
                </div>
            </div>
        </div>
        
        <div class="comparison-map-section">
            <h2>Comparação Visual das Rotas</h2>
            <div class="comparison-map-container">
                <div class="map" id="map_comparison">
                    <div id="map_comparison_leaflet">
                        {comparison_map_content}
                    </div>
                </div>
                <div class="comparison-legend">
                    <h3>Legenda</h3>
                    <div class="comparison-legend-item">
                        <div class="comparison-legend-color eco"></div>
                        <span>Rota Ecológica</span>
                    </div>
                    <div class="comparison-legend-item">
                        <div class="comparison-legend-color short"></div>
                        <span>Rota Mais Curta</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>Análise Comparativa: Vantagens e Desvantagens</h2>
            <div class="analysis-content">
                <div class="analysis-box eco">
                    <h3>Rota Ecológica</h3>
                    <ul>
                        <li><span class="advantage">✓ Vantagem:</span> Menor consumo de combustível ({comp['fuel_diff_liters']:+.3f}L de diferença)</li>
                        <li><span class="advantage">✓ Vantagem:</span> {'Menor tempo de viagem' if comp['time_diff_min'] < 0 else 'Tempo similar'}</li>
                        <li><span class="advantage">✓ Vantagem:</span> Mais sustentável e econômica a longo prazo</li>
                        <li><span class="disadvantage">✗ Desvantagem:</span> {'Distância ligeiramente maior' if comp['length_diff_m'] > 0 else 'Distância similar'} ({comp['length_diff_m']:+.1f}m)</li>
                    </ul>
                </div>
                
                <div class="analysis-box short">
                    <h3>Rota Mais Curta</h3>
                    <ul>
                        <li><span class="advantage">✓ Vantagem:</span> Menor distância percorrida ({comp['length_diff_m']:+.1f}m de diferença)</li>
                        <li><span class="advantage">✓ Vantagem:</span> Caminho mais direto entre origem e destino</li>
                        <li><span class="disadvantage">✗ Desvantagem:</span> Maior consumo de combustível ({comp['fuel_diff_liters']:+.3f}L a mais)</li>
                        <li><span class="disadvantage">✗ Desvantagem:</span> {'Tempo de viagem maior' if comp['time_diff_min'] > 0 else 'Tempo similar'} ({comp['time_diff_min']:+.1f} min)</li>
                    </ul>
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
    
    print(f"\nMapa comparativo salvo em: {out}")
    print(f"\nRota Ecológica: {result_eco['total_length_m']:.1f}m, {result_eco['total_fuel_liters']:.3f}L, {result_eco['total_time_min']:.1f}min")
    print(f"Rota Mais Curta: {result_short['total_length_m']:.1f}m, {result_short['total_fuel_liters']:.3f}L, {result_short['total_time_min']:.1f}min")
    print(f"\nComparação:")
    print(f"  Diferença de distância: {comp['length_diff_m']:+.1f} m ({comp['length_diff_pct']:+.1f}%)")
    print(f"  Diferença de combustível: {comp['fuel_diff_liters']:+.3f} L ({comp['fuel_diff_pct']:+.1f}%)")
    print(f"  Diferença de tempo: {comp['time_diff_min']:+.1f} min ({comp['time_diff_pct']:+.1f}%)")
    
    return out


def render_separate_routes_with_comparison(start_addr: str, dest_addr: str, 
                                           output_dir: str = ".", 
                                           zoom_start: int = 14) -> Tuple[Path, Path]:
    """
    Alias para render_both_routes_to_html que mantém compatibilidade.
    Retorna o mesmo arquivo duas vezes para manter a assinatura.
    """
    output_file = Path(output_dir) / "rotas_comparacao.html"
    result = render_both_routes_to_html(start_addr, dest_addr, str(output_file), zoom_start)
    return result, result


def render_route_to_html(start_addr: str, dest_addr: str, output_html: str = "rota.html", zoom_start: int = 14) -> Path:
    """Renderiza apenas a rota ecológica (mantém compatibilidade)."""
    G = build_graph_from_csv()
    result = route_ecological(G, start_addr, dest_addr)

    coords = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result['path_nodes']]
    if not coords:
        raise ValueError("Rota vazia, não há coordenadas para desenhar.")

    m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    folium.PolyLine(coords, color="blue", weight=5, opacity=0.9, tooltip="Rota Ecológica").add_to(m)
    folium.Marker(coords[0], icon=folium.Icon(color="green"), tooltip="Início").add_to(m)
    folium.Marker(coords[-1], icon=folium.Icon(color="red"), tooltip="Destino").add_to(m)

    out = Path(output_html).resolve()
    m.save(str(out))
    print(f"Mapa salvo em: {out}")
    return out