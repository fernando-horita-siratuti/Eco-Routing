from calculation.data_processing import gerar_csvs
from pathlib import Path
from calculation.render_map import render_all_routes_combined
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    # TO ADAPT FOR ANOTHER CITY: Change CSV filenames to match your city (e.g., "yourcity_nodes.csv")
    nodes_csv = data_dir / "divinopolis_nodes.csv"
    edges_csv = data_dir / "divinopolis_edges.csv"

    if nodes_csv.exists() and edges_csv.exists():
        print("Data processed successfully.")
    else:
        print("Generating data...")
        # TO ADAPT FOR ANOTHER CITY: Change default place parameter in gerar_csvs() function
        # Twice to improve precision of elevation data
        gerar_csvs()
        gerar_csvs()
        print("Data processed successfully. Cache warmed up.")

    # TO ADAPT FOR ANOTHER CITY: Replace addresses with addresses from your target city
    start_addr = "Avenida Amazonas, 277, Centro, Divinópolis, MG, Brasil"
    dest_addr = "Rua Padre Eustáquio, 716, Santa Rosa, Divinópolis, MG, Brasil"
  
    print("Rendering routes...")
    render_all_routes_combined(start_addr, dest_addr, output_html="rotas.html")
    print("Routes rendered successfully.")