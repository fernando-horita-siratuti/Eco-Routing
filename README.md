<div align="center">
  <img src="banner.png" alt="ECO-ROUTING Banner" width="100%">
</div>

# Ecological Route Optimization System

A Python-based route optimization system that compares ecological routes (fuel-efficient) versus shortest distance routes using Dijkstra and A* algorithms. The system generates interactive maps and scientific charts comparing route metrics including distance, fuel consumption and travel time.

<div align="center">

![Maintained](https://img.shields.io/badge/Maintained-yes-green.svg)
![Python](https://img.shields.io/badge/Python-3.9.6-blue.svg?logo=python&logoColor=white)
![Made for](https://img.shields.io/badge/Made%20for-VSCode-blue.svg?logo=visual-studio-code&logoColor=white)
![Contributions](https://img.shields.io/badge/contributions-welcome-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Ecological](https://img.shields.io/badge/Ecological-yes-green.svg)

</div>

---

## Scientific Paper

This project constitutes the practical implementation of the research presented in the paper **"Eco-Routing in Urban Networks: A Comparative Study of A\* and Dijkstra on Ecology-Aware and Fastest Objectives"**.

<div align="center">

[![Read the Paper](https://img.shields.io/badge/Read%20the%20Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](eco_routing.pdf)

</div>

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Basic Usage](#basic-usage)
- [Adapting for Another City](#adapting-for-another-city)
- [Project Structure](#project-structure)
- [Configurable Parameters](#configurable-parameters)
- [Complexity Analysis](#complexity-analysis)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)
- [Contact Us](#contact-us)

---

## Features

- **Dual Algorithm Support**: Implements both Dijkstra and A* algorithms for route calculation
- **Ecological Route Optimization**: Calculates fuel-efficient routes considering terrain slope and road speed limits
- **Shortest Route Calculation**: Finds the shortest distance path between two points
- **Interactive Visualization**: Generates HTML files with interactive maps (Folium) and scientific charts (Matplotlib)
- **Performance Comparison**: Compares execution times and route metrics between algorithms
- **Geocoding Support**: Automatic address geocoding with support for duplicate street names

## Requirements

- **Python**: 3.9.6 (required)
- **Operating System**: Windows, Linux, or macOS
- **Memory**: Minimum 4GB RAM recommended
- **Disk Space**: ~500MB for dependencies and generated data
- **Internet Connection**: Required for downloading map data and elevation information

## Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/fernando-horita-siratuti/Eco-Routing.git
cd Eco-Routing
```

### 2. Create Virtual Environment

Create a virtual environment in the project directory:

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

This will install:
- numpy==1.22.4
- pandas==1.5.3
- matplotlib==3.6.3
- networkx==2.8.8
- shapely==1.8.5
- scipy==1.10.1
- geopandas==0.12.2
- osmnx==1.2.2
- folium==0.14.0
- pyrosm==0.5.3
- geopy==2.3.0

### 5. Verify Installation

Verify that all packages are installed correctly:

```bash
python -c "import osmnx, networkx, folium, geopy; print('All packages installed successfully!')"
```

## Data Sources

The system uses three main data sources, all of which are **free and do not require API keys**:

### 1. OpenStreetMap (OSM) - Road Network Data

- **Source**: [OpenStreetMap](https://www.openstreetmap.org/)
- **Access Method**: OSMnx library (automatic download)
- **Documentation**: [OSMnx Documentation](https://osmnx.readthedocs.io/)
- **Data Type**: Road network (drive network) - nodes and edges
- **Format**: Downloaded automatically via OSMnx, no manual download required
- **What it provides**: 
  - Street network topology
  - Road names and types
  - Speed limits
  - One-way restrictions
  - Road lengths

### 2. Open-Elevation API - Elevation Data

- **Source**: [Open-Elevation API](https://api.open-elevation.com/)
- **Type**: Free REST API
- **API Key**: Not required
- **Rate Limiting**: None (but implemented with batch requests for efficiency)
- **What it provides**: Elevation data (in meters) for any coordinate pair
- **Usage**: Automatically fetched during data processing, cached locally

### 3. Nominatim (OpenStreetMap) - Geocoding Service

- **Source**: [Nominatim](https://nominatim.openstreetmap.org/)
- **Type**: Free geocoding service (part of OpenStreetMap)
- **API Key**: Not required
- **Rate Limiting**: 1 request per second (already implemented in code)
- **Usage Policy**: [Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)
- **What it provides**: 
  - Address to coordinates conversion (geocoding)
  - Coordinates to address conversion (reverse geocoding)

## Basic Usage

### Running the Program

Once the environment is set up and dependencies are installed, run the program:

```bash
python -m main.main
```

Or from the project root:

```bash
python main/main.py
```

If the previous commands do not work, try using:

```bash
python3 -m main.main
```

### What to Expect

1. **First Run**: 
   - The program will download road network data from OpenStreetMap (this may take 5-15 minutes depending on city size)
   - Elevation data will be fetched from Open-Elevation API
   - CSV files will be generated in the `data/` directory
   - Total time: 10-20 minutes for a medium-sized city

2. **Subsequent Runs**:
   - If CSV files already exist, data processing is skipped
   - Route calculation and rendering: 1-3 minutes
   - Geocoding: 2-5 seconds per address

3. **Output**:
   - An HTML file named `rotas.html` will be generated in the project root
   - Open this file in a web browser to view interactive maps and charts

### Example Output

The generated HTML file contains:
- Interactive maps showing ecological and shortest routes
- Comparison charts for distance, fuel consumption and travel time
- Performance comparison between Dijkstra and A* algorithms
- Route statistics and metrics

## Adapting for Another City

To use this system with a different city, follow these steps:

### Step 1: Modify `calculation/data_processing.py`

1. **Change the default place parameter** in the `gerar_csvs()` function (line 61):

```python
def gerar_csvs(place: str = "Your City, State, Country") -> Tuple[Path, Path]:
```

Replace `"Your City, State, Country"` with your target city in OSMnx format. Examples:
- `"New York, NY, USA"`
- `"London, UK"`
- `"São Paulo, SP, Brazil"`

2. **Update CSV filenames** (lines 113 and 130):

```python
# Line 113
nodes_csv = OUTPUT_DIR / "yourcity_nodes.csv"

# Line 130
edges_csv = OUTPUT_DIR / "yourcity_edges.csv"
```

### Step 2: Modify `calculation/dijkstra.py`

1. **Update neighborhood list** in `geocode_address()` function (lines 300-308):

```python
# TO ADAPT FOR ANOTHER CITY: Replace this list with neighborhoods/districts of your target city
neighborhoods_divinopolis = [
    "Neighborhood1", "Neighborhood2", "Neighborhood3",
    # ... add all relevant neighborhoods for your city
]
```

2. **Update street-to-neighborhood mapping** (lines 327-334):

```python
# TO ADAPT FOR ANOTHER CITY: Update this dictionary with streets that have duplicates in your city
street_to_bairro = {
    "street_name_1": ["Neighborhood1", "Neighborhood2"],
    "street_name_2": ["Neighborhood3", "Neighborhood4"],
    # ... add streets with multiple locations
}
```

3. **Update city location strings** throughout the function:
   - Replace all occurrences of `"Divinópolis, MG, Brasil"` with your city's location string
   - Search for: `"Divinópolis"` and replace with your city name

4. **Adjust regex patterns** (if your language uses different street prefixes, line 315):

```python
# TO ADAPT FOR ANOTHER LANGUAGE: Modify regex patterns to match your language's street prefixes
street_match = re.search(r'(rua|avenida|av\.?|r\.?|street|avenue|st|ave)\s+([^,0-9]+?)(?:\s*,\s*|\s*$)', address_lower)
```

### Step 3: Modify `main/main.py`

1. **Update CSV filenames** (lines 10-11):

```python
# TO ADAPT FOR ANOTHER CITY: Change CSV filenames to match your city
nodes_csv = data_dir / "yourcity_nodes.csv"
edges_csv = data_dir / "yourcity_edges.csv"
```

2. **Update example addresses** (lines 21-22):

```python
# TO ADAPT FOR ANOTHER CITY: Replace addresses with addresses from your target city
# Format: "Street Name, Number, Neighborhood, City, State, Country"
start_addr = "Main Street, 123, Downtown, Your City, State, Country"
dest_addr = "Second Avenue, 456, Uptown, Your City, State, Country"
```

### Step 4: Generate New Data

1. **Delete old CSV files** (if they exist):
   ```bash
   rm data/divinopolis_nodes.csv data/divinopolis_edges.csv
   ```

2. **Run the program** to generate new data:
   ```bash
   python -m main.main
   ```

3. **Verify data generation**:
   - Check that new CSV files were created in the `data/` directory
   - Verify file sizes are reasonable (should be several MB for a medium city)

### Step 5: Test Geocoding

Test that addresses are being geocoded correctly:

1. Use addresses with neighborhood information for better accuracy
2. If geocoding fails or returns incorrect locations:
   - Add more neighborhoods to the list in `geocode_address()`
   - Update the `street_to_bairro` mapping for problematic streets
   - Check that address format matches your city's conventions

## Project Structure

```
TrabFinalAEDS2/
├── calculation/
│   ├── a_star.py              # A* algorithm implementation
│   ├── data_processing.py      # OSM data download and CSV generation
│   ├── dijkstra.py             # Dijkstra algorithm and geocoding
│   ├── elevation.py            # Elevation data fetching and calculations
│   ├── render_map.py           # HTML map and chart generation
│   └── elevation_cache.json   # Cached elevation data
├── data/
│   ├── divinopolis_nodes.csv   # Graph nodes (generated)
│   ├── divinopolis_edges.csv  # Graph edges (generated)
│   └── sudesteMap.osm.pbf     # Optional: OSM PBF file
├── main/
│   └── main.py                 # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── rotas.html                  # Generated output (after running)
```

### Key Files Description

- **`calculation/dijkstra.py`**: Core routing algorithms, graph building and geocoding
- **`calculation/a_star.py`**: A* algorithm implementation with heuristics
- **`calculation/data_processing.py`**: Downloads OSM data and generates CSV files
- **`calculation/elevation.py`**: Fetches and caches elevation data from Open-Elevation API
- **`calculation/render_map.py`**: Creates interactive HTML maps and scientific charts
- **`main/main.py`**: Main script that orchestrates the entire process

## Configurable Parameters

The following parameters can be adjusted in `calculation/dijkstra.py` (lines 10-15) to match different vehicle characteristics or local conditions:

```python
BASE_L_PER_100KM = 10.0       # Typical base consumption (L/100km) at moderate speed
SLOPE_COEF = 10.0             # How much uphill increases consumption (multiplier per unit of slope)
SPEED_PENALTY_COEF = 0.2      # Penalty for speeds outside reference (quadratic)
REF_SPEED_KMH = 50.0          # Reference speed for consumption (km/h)
TIME_WEIGHT = 0.5             # How many "equivalent liters" assigned to 1 extra minute
```

**To adapt these parameters:**
- `BASE_L_PER_100KM`: Adjust based on your vehicle's fuel efficiency
- `SLOPE_COEF`: Increase for vehicles more sensitive to inclines
- `REF_SPEED_KMH`: Set to typical city driving speed in your region
- `TIME_WEIGHT`: Balance between fuel consumption and time savings

## Complexity Analysis

This project implements two pathfinding algorithms: **Dijkstra** and **A\***. Both algorithms have time complexity **O(E log V)** where E is the number of edges and V is the number of vertices. However, A* is more efficient in practice because it uses heuristics to guide the search toward the destination, exploring fewer nodes.

### Key Points:

- **Dijkstra**: Explores all nodes within a radius from the start point
- **A\***: Uses admissible heuristics to focus search on optimal path, exploring ~30-50% fewer nodes
- **Space Complexity**: Both algorithms use O(V) space
- **Heuristics**: A* uses haversine distance (for shortest path) and ecological cost lower bounds (for eco route)

For detailed complexity analysis, see [docs/ANALYSIS.md](docs/ANALYSIS.md).

## Technical Details

### Heuristics

The A* algorithm uses **admissible heuristics** that never overestimate the actual cost:

1. **Shortest Distance Heuristic**: Uses haversine (straight-line) distance
   - Always ≤ actual path distance
   - Guarantees optimality

2. **Ecological Cost Heuristic**: Uses lower bounds for fuel and time
   - Assumes ideal conditions (no slope, maximum speed)
   - Never overestimates actual ecological cost

### Cost Formula

The ecological cost combines fuel consumption and time penalty:

```
eco_cost = fuel_liters + time_penalty_equiv_liters

where:
fuel_liters = base_per_m × length × slope_multiplier × speed_factor
time_penalty = TIME_WEIGHT × time_minutes × liters_per_min_ref
```

### Inclination Calculation

Street inclination is calculated using:

```
grade = dh / dist_horizontal
inclination_deg = atan2(dh, dist_horizontal) × (180/π)
```

For comprehensive technical documentation, see [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Testing

The project includes unit tests to validate algorithm correctness and heuristics admissibility.

### Running Tests

To run the test suite:

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest tests.test_algorithms
```

### Test Coverage

The test suite includes:
- **Haversine distance calculation**: Validates geographic distance computations
- **Dijkstra manual implementation**: Tests pathfinding correctness
- **A* heuristics**: Verifies admissibility (never overestimate)
- **Street steepness**: Validates inclination calculations
- **Algorithm consistency**: Ensures Dijkstra manual matches NetworkX results

See `tests/test_algorithms.py` for complete test implementations.

## Troubleshooting

### Problem: Geocoding returns incorrect addresses

**Solution:**
1. Add the correct neighborhood to the `neighborhoods_divinopolis` list in `geocode_address()`
2. For streets with duplicate names, add them to `street_to_bairro` mapping
3. Ensure addresses include neighborhood information when possible
4. Check that the address format matches your city's conventions

### Problem: "No path between selected nodes"

**Solution:**
1. Verify that both addresses were geocoded correctly
2. Check that the road network data was downloaded completely
3. Ensure the addresses are within the downloaded city boundaries
4. Try addresses that are closer together to test connectivity

### Problem: Data download takes too long

**Solution:**
1. For very large cities, consider downloading a smaller region
2. Use OSMnx's `network_type` parameter to limit road types
3. Check internet connection speed
4. The first download is always slower; subsequent runs use cached data

### Problem: Elevation data not found

**Solution:**
1. Check internet connection (elevation data is fetched from API)
2. Verify Open-Elevation API is accessible: https://api.open-elevation.com/
3. Check `calculation/elevation_cache.json` for cached data
4. The system will continue with elevation=0 if API fails (routes will still work)

### Problem: Import errors

**Solution:**
1. Ensure virtual environment is activated
2. Verify all packages are installed: `pip list`
3. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
4. Check Python version: `python --version` (should be 3.9.6)

### Problem: CSV files not found

**Solution:**
1. Run the program once to generate CSV files
2. Check that `data/` directory exists
3. Verify write permissions in the project directory
4. Check console output for error messages during data generation

## References

### Libraries Documentation

- **OSMnx**: [https://osmnx.readthedocs.io/](https://osmnx.readthedocs.io/)
- **NetworkX**: [https://networkx.org/documentation/](https://networkx.org/documentation/)
- **Folium**: [https://python-visualization.github.io/folium/](https://python-visualization.github.io/folium/)
- **Geopy**: [https://geopy.readthedocs.io/](https://geopy.readthedocs.io/)
- **Matplotlib**: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

### Data Sources

- **OpenStreetMap**: [https://www.openstreetmap.org/](https://www.openstreetmap.org/)
- **Open-Elevation API**: [https://api.open-elevation.com/](https://api.open-elevation.com/)
- **Nominatim**: [https://nominatim.openstreetmap.org/](https://nominatim.openstreetmap.org/)

### Algorithm References

- **Dijkstra's Algorithm**: 
  - Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. Chapter 24: Single-Source Shortest Paths.

- **A\* Algorithm**: 
  - Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. (Heuristic search algorithms).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Us

<div align="center">
  
  <br><br>
     <i>Fernando Horita Siratuti - Undergraduate - 4th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor1]
  [![Linkedin][linkedin-badge]][linkedin-autor1]
  [![GitHub][github-badge]][github-autor1]
  [![Instagram][instagram-badge]][instagram-autor1]
  
  <br><br>
     <i>Hugo Henrique Marques - Undergraduate - 4th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor2]
  [![Linkedin][linkedin-badge]][linkedin-autor2]
  [![GitHub][github-badge]][github-autor2]
  [![Instagram][instagram-badge]][instagram-autor2]

</div>

[gmail-badge]: https://img.shields.io/badge/-Gmail-c14438?style=flat-square&logo=Gmail&logoColor=white
[linkedin-badge]: https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white
[github-badge]: https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white
[instagram-badge]: https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white

[gmail-autor1]: mailto:siratutifernando@gmail.com
[linkedin-autor1]: https://www.linkedin.com/in/fernando-horita-siratuti/
[github-autor1]: https://github.com/fernando-horita-siratuti
[instagram-autor1]: https://www.instagram.com/siratuti_/

[gmail-autor2]: mailto:hugohmarques4@gmail.com
[linkedin-autor2]: https://www.linkedin.com/in/hugo-h-marques-980629216/
[github-autor2]: https://github.com/hugnarok
[instagram-autor2]: https://www.instagram.com/hugomarques_02/
