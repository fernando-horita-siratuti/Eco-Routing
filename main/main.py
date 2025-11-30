from calculation.data_processing import gerar_csvs
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
        print("Dados tratados com sucesso.")
    else:
        print("Gerando dados...")
        gerar_csvs()
        print("Dados tratados com sucesso.")

    start_addr = "Rua Padre Eustáquio, 710, Divinópolis, MG, Brasil"
    dest_addr = "Álvares de Azevedo, 400, Divinópolis, MG, Brasil"
  
    print("Renderizando rotas...")
    render_all_routes_combined(start_addr, dest_addr, output_html="rotas.html")
    print("Rotas renderizadas com sucesso.")