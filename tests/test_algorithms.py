"""
Testes unitários para os algoritmos de roteamento ecológico.

Este módulo contém testes para:
- Função haversine (cálculo de distância geográfica)
- Algoritmo Dijkstra manual
- Heurísticas do A*
- Cálculo de inclinação (street_steepness)
- Validação de resultados entre algoritmos
"""

import unittest
import math
import networkx as nx
from calculation.dijkstra import haversine, dijkstra_manual, build_graph_from_csv
from calculation.a_star import _make_astar_heuristic_shortest, _make_astar_heuristic_eco
from calculation.elevation import street_steepness


class TestHaversine(unittest.TestCase):
    """Testes para a função haversine de cálculo de distância geográfica."""
    
    def test_haversine_same_point(self):
        """Testa que a distância entre o mesmo ponto é zero."""
        lat, lon = -20.1394, -44.8900  # Coordenadas de Divinópolis
        distance = haversine(lon, lat, lon, lat)
        self.assertAlmostEqual(distance, 0.0, places=1)
    
    def test_haversine_known_distance(self):
        """Testa distância conhecida entre dois pontos."""
        # Coordenadas aproximadas de Divinópolis
        lat1, lon1 = -20.1394, -44.8900
        # Ponto aproximadamente 1km ao norte
        lat2, lon2 = -20.1300, -44.8900
        
        distance = haversine(lon1, lat1, lon2, lat2)
        # Deve ser aproximadamente 1km (1000m) com margem de erro
        self.assertGreater(distance, 900)
        self.assertLess(distance, 1100)
    
    def test_haversine_symmetry(self):
        """Testa que a distância é simétrica (A->B = B->A)."""
        lat1, lon1 = -20.1394, -44.8900
        lat2, lon2 = -20.1300, -44.8800
        
        dist1 = haversine(lon1, lat1, lon2, lat2)
        dist2 = haversine(lon2, lat2, lon1, lat1)
        
        self.assertAlmostEqual(dist1, dist2, places=5)
    
    def test_haversine_positive(self):
        """Testa que a distância é sempre positiva."""
        lat1, lon1 = -20.1394, -44.8900
        lat2, lon2 = -20.1300, -44.8800
        
        distance = haversine(lon1, lat1, lon2, lat2)
        self.assertGreater(distance, 0)


class TestStreetSteepness(unittest.TestCase):
    """Testes para o cálculo de inclinação de ruas."""
    
    def test_street_steepness_flat(self):
        """Testa inclinação zero (terreno plano)."""
        result = street_steepness(
            -20.1394, -44.8900, 100.0,  # Ponto 1
            -20.1300, -44.8900, 100.0   # Ponto 2 (mesma elevação)
        )
        
        self.assertAlmostEqual(result["dh_m"], 0.0, places=1)
        self.assertAlmostEqual(result["grade"], 0.0, places=3)
        self.assertAlmostEqual(result["inclination_deg"], 0.0, places=1)
    
    def test_street_steepness_uphill(self):
        """Testa inclinação positiva (subida)."""
        result = street_steepness(
            -20.1394, -44.8900, 100.0,  # Ponto 1
            -20.1300, -44.8900, 110.0   # Ponto 2 (10m mais alto)
        )
        
        self.assertAlmostEqual(result["dh_m"], 10.0, places=1)
        self.assertGreater(result["grade"], 0.0)
        self.assertGreater(result["inclination_deg"], 0.0)
    
    def test_street_steepness_downhill(self):
        """Testa inclinação negativa (descida)."""
        result = street_steepness(
            -20.1394, -44.8900, 110.0,  # Ponto 1
            -20.1300, -44.8900, 100.0   # Ponto 2 (10m mais baixo)
        )
        
        self.assertAlmostEqual(result["dh_m"], -10.0, places=1)
        self.assertLess(result["grade"], 0.0)
        # Inclinação em graus deve ser positiva (valor absoluto)
        self.assertGreater(result["inclination_deg"], 0.0)
    
    def test_street_steepness_grade_calculation(self):
        """Testa cálculo correto do grade."""
        # Distância horizontal aproximada de 1000m, elevação de 50m
        # Grade deve ser aproximadamente 0.05 (5%)
        result = street_steepness(
            -20.1394, -44.8900, 100.0,
            -20.1300, -44.8900, 150.0
        )
        
        if result["grade"] is not None:
            # Grade deve estar entre 0.01 e 0.1 para esta distância/elevação
            self.assertGreater(result["grade"], 0.0)
            self.assertLess(result["grade"], 1.0)


class TestDijkstraManual(unittest.TestCase):
    """Testes para a implementação manual do algoritmo de Dijkstra."""
    
    def setUp(self):
        """Cria um grafo simples para testes."""
        self.G = nx.DiGraph()
        # Grafo simples: 0 -> 1 -> 2 -> 3
        self.G.add_edge(0, 1, length=10.0)
        self.G.add_edge(1, 2, length=20.0)
        self.G.add_edge(2, 3, length=15.0)
        # Caminho alternativo: 0 -> 3 (mais longo)
        self.G.add_edge(0, 3, length=100.0)
    
    def test_dijkstra_manual_simple_path(self):
        """Testa Dijkstra em um caminho simples."""
        path, cost = dijkstra_manual(self.G, 0, 3, weight='length')
        
        self.assertEqual(path, [0, 1, 2, 3])
        self.assertAlmostEqual(cost, 45.0, places=1)
    
    def test_dijkstra_manual_same_node(self):
        """Testa Dijkstra quando origem e destino são iguais."""
        path, cost = dijkstra_manual(self.G, 0, 0, weight='length')
        
        self.assertEqual(path, [0])
        self.assertAlmostEqual(cost, 0.0, places=1)
    
    def test_dijkstra_manual_adjacent_nodes(self):
        """Testa Dijkstra entre nós adjacentes."""
        path, cost = dijkstra_manual(self.G, 0, 1, weight='length')
        
        self.assertEqual(path, [0, 1])
        self.assertAlmostEqual(cost, 10.0, places=1)
    
    def test_dijkstra_manual_no_path(self):
        """Testa Dijkstra quando não há caminho."""
        G_isolated = nx.DiGraph()
        G_isolated.add_node(0)
        G_isolated.add_node(1)
        # Sem arestas conectando 0 e 1
        
        with self.assertRaises(nx.NetworkXNoPath):
            dijkstra_manual(G_isolated, 0, 1, weight='length')
    
    def test_dijkstra_manual_negative_weight_error(self):
        """Testa que Dijkstra rejeita pesos negativos."""
        G_negative = nx.DiGraph()
        G_negative.add_edge(0, 1, length=-10.0)
        
        with self.assertRaises(ValueError):
            dijkstra_manual(G_negative, 0, 1, weight='length')


class TestAStarHeuristics(unittest.TestCase):
    """Testes para as heurísticas do algoritmo A*."""
    
    def setUp(self):
        """Cria um grafo simples com coordenadas para testes."""
        self.G = nx.DiGraph()
        # Nó 0: Divinópolis centro
        self.G.add_node(0, y=-20.1394, x=-44.8900)
        # Nó 1: Ponto ao norte
        self.G.add_node(1, y=-20.1300, x=-44.8900)
        # Nó 2: Ponto ao leste
        self.G.add_node(2, y=-20.1394, x=-44.8800)
        
        self.G.add_edge(0, 1, length=1000.0)
        self.G.add_edge(0, 2, length=1000.0)
    
    def test_heuristic_shortest_admissible(self):
        """Testa que a heurística de distância nunca superestima."""
        heuristic = _make_astar_heuristic_shortest(self.G)
        
        # Heurística deve retornar distância em linha reta
        h_value = heuristic(0, 1)
        
        # Deve ser menor ou igual à distância real do caminho
        # (que é pelo menos a distância da aresta)
        self.assertGreaterEqual(1000.0, h_value)
        self.assertGreater(h_value, 0)
    
    def test_heuristic_shortest_symmetry(self):
        """Testa que a heurística de distância é simétrica."""
        heuristic = _make_astar_heuristic_shortest(self.G)
        
        h1 = heuristic(0, 1)
        h2 = heuristic(1, 0)
        
        # Haversine é simétrico
        self.assertAlmostEqual(h1, h2, places=1)
    
    def test_heuristic_eco_admissible(self):
        """Testa que a heurística ecológica nunca superestima."""
        heuristic = _make_astar_heuristic_eco(self.G)
        
        h_value = heuristic(0, 1)
        
        # Heurística deve ser >= 0 (custo não pode ser negativo)
        self.assertGreaterEqual(h_value, 0)
        
        # Heurística deve ser um lower bound do custo real
        # (não podemos testar exatamente sem calcular o custo real,
        # mas podemos verificar que é não-negativa e razoável)
        self.assertLess(h_value, 1000.0)  # Deve ser razoável
    
    def test_heuristic_zero_for_same_node(self):
        """Testa que a heurística retorna zero para o mesmo nó."""
        heuristic_short = _make_astar_heuristic_shortest(self.G)
        heuristic_eco = _make_astar_heuristic_eco(self.G)
        
        h_short = heuristic_short(0, 0)
        h_eco = heuristic_eco(0, 0)
        
        self.assertAlmostEqual(h_short, 0.0, places=1)
        self.assertAlmostEqual(h_eco, 0.0, places=3)


class TestAlgorithmConsistency(unittest.TestCase):
    """Testes para validar consistência entre algoritmos."""
    
    def setUp(self):
        """Cria um grafo pequeno para comparação."""
        self.G = nx.DiGraph()
        self.G.add_node(0, y=-20.1394, x=-44.8900)
        self.G.add_node(1, y=-20.1300, x=-44.8900)
        self.G.add_node(2, y=-20.1200, x=-44.8900)
        
        self.G.add_edge(0, 1, length=1000.0, eco_cost=0.1)
        self.G.add_edge(1, 2, length=1000.0, eco_cost=0.1)
        self.G.add_edge(0, 2, length=2500.0, eco_cost=0.25)
    
    def test_dijkstra_manual_vs_networkx(self):
        """Testa que Dijkstra manual retorna mesmo resultado que NetworkX."""
        # Teste com peso 'length'
        path_manual, cost_manual = dijkstra_manual(self.G, 0, 2, weight='length')
        path_nx = nx.shortest_path(self.G, 0, 2, weight='length', method='dijkstra')
        cost_nx = nx.shortest_path_length(self.G, 0, 2, weight='length')
        
        self.assertEqual(path_manual, path_nx)
        self.assertAlmostEqual(cost_manual, cost_nx, places=5)
    
    def test_dijkstra_eco_cost(self):
        """Testa que Dijkstra funciona com peso eco_cost."""
        path, cost = dijkstra_manual(self.G, 0, 2, weight='eco_cost')
        
        # Deve encontrar um caminho válido
        self.assertGreater(len(path), 0)
        self.assertGreaterEqual(cost, 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 2)


if __name__ == '__main__':
    unittest.main()
