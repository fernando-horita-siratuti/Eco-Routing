from calculation.data_processing import gerar_csvs
from calculation.routing import calculate_route
from pathlib import Path
from calculation.render_map import render_both_routes_to_html

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
    print(f"Rota Calculada com sucesso\n")

    start_addr = "Rua Padre Eustáquio, 716, Divinópolis, MG, Brasil"
    dest_addr = "Rua Rio de Janeiro, 2220, Divinópolis, MG, Brasil"
  
    render_both_routes_to_html(start_addr, dest_addr, output_html="rotas_comparacao.html")