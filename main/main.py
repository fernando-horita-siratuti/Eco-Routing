from calculation.data_processing import gerar_csvs
from calculation.dijkstra import calculate_route
from pathlib import Path
from calculation.render_map import render_all_routes_combined
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    nodes_csv = data_dir / "divinopolis_nodes.csv"
    edges_csv = data_dir / "divinopolis_edges.csv"

    if nodes_csv.exists() and edges_csv.exists():
        print(f"Dados Tratados com sucesso")
    else:
        gerar_csvs()
        print(f"Dados Tratados com sucesso")

    calculate_route()
    print(f"Rota Calculada com sucesso")

    start_addr = "Rua Padre Eustáquio, 710, Divinópolis, MG, Brasil"
    dest_addr = "Álvares de Azevedo, 400, Divinópolis, MG, Brasil"
  
    # Renderiza todas as rotas (Dijkstra + A*) em um único arquivo
    print(f"Renderizando rotas")
    render_all_routes_combined(start_addr, dest_addr, output_html="rotas.html")