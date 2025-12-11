"""
Unit test package for the ecological routing project.

This package contains tests to validate the correctness and efficiency of implemented
algorithms, including Dijkstra, A*, geographic calculations, and routing metrics.

Usage examples:

1. Run all tests:
   ```bash
   python -m pytest tests/
   ```

2. Run specific test module:
   ```bash
   python -m unittest tests.test_algorithms
   ```

3. Run with verbose output (shows details):
   ```bash
   python -m pytest tests/ -v
   ```

4. Run a specific test:
   ```bash
   python -m unittest tests.test_algorithms.TestHaversine.test_haversine_same_point
   ```

5. Run with code coverage:
   ```bash
   python -m pytest tests/ --cov=calculation
   ```

Test modules:
- test_algorithms.py: Tests for routing algorithms (Dijkstra, A*), 
  geographic calculations (Haversine) and metrics (slope, ecological cost)
"""

__version__ = "1.0.0"
__author__ = "Final Project AED2"

# Common imports to facilitate usage in tests
import unittest
import sys
from pathlib import Path

# Add root directory to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Test configuration example (can be expanded)
class TestConfig:
    """Configuration for tests."""
    # Test coordinates (Divinópolis, MG)
    TEST_LAT = -20.1394
    TEST_LON = -44.8900
    
    # Tolerance for floating point comparisons
    FLOAT_TOLERANCE = 1e-5
    
    # Minimum graph size for tests
    MIN_GRAPH_SIZE = 3

# Helper function example for tests
def create_simple_test_graph():
    """
    Creates a simple graph for testing.
    
    Returns:
        networkx.DiGraph: Simple directed graph with 4 nodes and 3 edges
    """
    import networkx as nx
    
    G = nx.DiGraph()
    # Linear graph: 0 -> 1 -> 2 -> 3
    G.add_edge(0, 1, length=10.0, eco_cost=0.1)
    G.add_edge(1, 2, length=20.0, eco_cost=0.2)
    G.add_edge(2, 3, length=15.0, eco_cost=0.15)
    
    # Add coordinates to nodes (required for some tests)
    G.nodes[0]['y'] = -20.1394
    G.nodes[0]['x'] = -44.8900
    G.nodes[1]['y'] = -20.1300
    G.nodes[1]['x'] = -44.8900
    G.nodes[2]['y'] = -20.1200
    G.nodes[2]['x'] = -44.8900
    G.nodes[3]['y'] = -20.1100
    G.nodes[3]['x'] = -44.8900
    
    return G

# Base test class example (can be inherited)
class BaseTestCase(unittest.TestCase):
    """Base test class with common utilities."""
    
    def assertAlmostEqualList(self, list1, list2, places=7, msg=None):
        """
        Checks if two lists of numbers are approximately equal.
        
        Args:
            list1: First list
            list2: Second list
            places: Number of decimal places for comparison
            msg: Custom error message
        """
        self.assertEqual(len(list1), len(list2), 
                        msg=f"Lists have different sizes: {len(list1)} vs {len(list2)}")
        for i, (a, b) in enumerate(zip(list1, list2)):
            self.assertAlmostEqual(a, b, places=places, 
                                  msg=f"Element {i} differs: {a} vs {b}")

# Decorator example for tests that require external data
def requires_data(func):
    """
    Decorator to mark tests that require external data (CSV, API, etc.).
    
    Example:
        @requires_data
        def test_with_real_data(self):
            # Test that needs real data
            pass
    """
    def wrapper(*args, **kwargs):
        # Here we could check if data exists
        # For now, just execute the test
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# Pytest fixture example (can be used with pytest)
def pytest_configure(config):
    """Pytest configuration (if using pytest)."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests that take long to run"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
