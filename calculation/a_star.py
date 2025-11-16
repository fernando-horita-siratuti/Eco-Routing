import networkx as nx
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Optional

def a_star(G: nx.DiGraph, start: int, target: int, heuristic: str = 'manhattan') -> List[int]:
    """
    Implementa o algoritmo A* para encontrar o caminho mais curto em um grafo.
    """
    if heuristic not in ['manhattan', 'euclidean']:
        raise ValueError("Heurística inválida. Use 'manhattan' ou 'euclidean'.")
    
    # Inicialização
    frontier = []
    heappush(frontier, (0, start))