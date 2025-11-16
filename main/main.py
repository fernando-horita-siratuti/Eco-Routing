from calculation.data_processing import gerar_csvs
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    nodes_csv = data_dir / "divinopolis_nodes.csv"
    edges_csv = data_dir / "divinopolis_edges.csv"

    if nodes_csv.exists() and edges_csv.exists():
        print(f"Dados Tratados com sucesso")
    else:
        gerar_csvs()
        print(f"Dados Tratados com sucesso")

    