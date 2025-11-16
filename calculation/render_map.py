# /Users/hugohenriquemarques/Desktop/Trabalho Final AED2/Repositorio/TrabFinalAEDS2/calculation/render_map.py
from pathlib import Path
import folium
from .routing import build_graph_from_csv, route_ecological, route_shortest_distance

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


def render_both_routes_to_html(start_addr: str, dest_addr: str, output_html: str = "rotas_comparacao.html", zoom_start: int = 14) -> Path:
    """
    Renderiza ambas as rotas (ecológica e mais curta) no mesmo mapa com cores diferentes.
    
    Args:
        start_addr: Endereço de origem
        dest_addr: Endereço de destino
        output_html: Nome do arquivo HTML de saída
        zoom_start: Nível de zoom inicial do mapa
    
    Returns:
        Path do arquivo HTML gerado
    """
    G = build_graph_from_csv()
    
    # Calcula ambas as rotas
    print("Calculando rota ecológica...")
    result_eco = route_ecological(G, start_addr, dest_addr)
    
    print("Calculando rota mais curta...")
    result_short = route_shortest_distance(G, start_addr, dest_addr)
    
    # Extrai coordenadas das rotas
    coords_eco = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_eco['path_nodes']]
    coords_short = [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in result_short['path_nodes']]
    
    if not coords_eco or not coords_short:
        raise ValueError("Uma das rotas está vazia, não há coordenadas para desenhar.")
    
    # Centraliza no ponto de início
    m = folium.Map(location=[coords_eco[0][0], coords_eco[0][1]], zoom_start=zoom_start, tiles="CartoDB positron")
    
    # Desenha rota ecológica (azul)
    folium.PolyLine(
        coords_eco, 
        color="blue", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Rota Ecológica - {result_eco['total_length_m']:.0f}m, {result_eco['total_fuel_liters']:.3f}L"
    ).add_to(m)
    
    # Desenha rota mais curta (vermelho)
    folium.PolyLine(
        coords_short, 
        color="red", 
        weight=5, 
        opacity=0.8,
        tooltip=f"Rota Mais Curta - {result_short['total_length_m']:.0f}m, {result_short['total_fuel_liters']:.3f}L"
    ).add_to(m)
    
    # Marcadores de início e fim
    folium.Marker(coords_eco[0], icon=folium.Icon(color="green"), tooltip="Início").add_to(m)
    folium.Marker(coords_eco[-1], icon=folium.Icon(color="red"), tooltip="Destino").add_to(m)
    
    # Adiciona legenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 100px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Legenda:</b></p>
    <p><i class="fa fa-circle" style="color:blue"></i> Rota Ecológica</p>
    <p><i class="fa fa-circle" style="color:red"></i> Rota Mais Curta</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    out = Path(output_html).resolve()
    m.save(str(out))
    print(f"\nMapa comparativo salvo em: {out}")
    print(f"\nRota Ecológica: {result_eco['total_length_m']:.1f}m, {result_eco['total_fuel_liters']:.3f}L")
    print(f"Rota Mais Curta: {result_short['total_length_m']:.1f}m, {result_short['total_fuel_liters']:.3f}L")
    
    return out