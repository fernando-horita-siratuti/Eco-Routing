import json
import math
import threading
import time
import requests
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_CACHE_LOCK = threading.Lock()
_CACHE_FILE = Path(__file__).resolve().parent / "elevation_cache.json"
_CACHE: Dict[str, float] = {}
_SESSION = requests.Session()
_SESSION_TIMEOUT = 5
_MAX_BATCH_SIZE = 50
_RETRY_ATTEMPTS = 3
_RETRY_SLEEP = 1.0

EARTH_RADIUS_M = 6371000.0


def _load_cache() -> None:
    """Loads elevation cache from file."""
    if _CACHE:
        return
    if not _CACHE_FILE.exists():
        return
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return
    if isinstance(data, dict):
        with _CACHE_LOCK:
            for key, value in data.items():
                try:
                    _CACHE[key] = float(value)
                except (TypeError, ValueError):
                    continue


def _save_cache() -> None:
    """Saves elevation cache to file."""
    with _CACHE_LOCK:
        snapshot = dict(_CACHE)
        tmp_path = _CACHE_FILE.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, ensure_ascii=False)
        tmp_path.replace(_CACHE_FILE)


def _coord_key(lat: float, lon: float, precision: int = 6) -> str:
    """Generates cache key for coordinates."""
    return f"{round(lat, precision)},{round(lon, precision)}"


def _batched(iterable: Iterable[Tuple[float, float]], size: int) -> Iterable[List[Tuple[float, float]]]:
    """
    Divides an iterable into batches of specified size.
    
    Args:
        iterable: Iterable of coordinates
        size: Size of each batch
    
    Yields:
        Lists of coordinates (batches)
    """
    batch: List[Tuple[float, float]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fetch_batch(coords: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """
    Fetches elevations for a batch of coordinates via API.
    
    Args:
        coords: List of coordinates (lat, lon)
    
    Returns:
        Dictionary mapping coordinate keys to elevations
    """
    locations = "|".join(f"{lat},{lon}" for lat, lon in coords)
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

    for attempt in range(_RETRY_ATTEMPTS):
        fetched: Dict[str, Optional[float]] = {}
        try:
            response = _SESSION.get(url, timeout=_SESSION_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
        except Exception:
            results = None

        if results:
            for item in results:
                lat = item.get("latitude")
                lon = item.get("longitude")
                elevation = item.get("elevation")
                if lat is None or lon is None:
                    continue
                fetched[_coord_key(lat, lon)] = elevation

        missing_keys = [
            _coord_key(lat, lon)
            for lat, lon in coords
            if _coord_key(lat, lon) not in fetched
        ]

        if fetched and not missing_keys:
            return fetched

        if attempt < _RETRY_ATTEMPTS - 1:
            time.sleep(_RETRY_SLEEP * (attempt + 1))

    fallback: Dict[str, Optional[float]] = {}
    for lat, lon in coords:
        fallback[_coord_key(lat, lon)] = _fetch_single(lat, lon)
    return fallback


def _fetch_single(lat: float, lon: float) -> Optional[float]:
    """
    Fetches elevation for a single coordinate via API.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Elevation in meters or None if not found
    """
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = _SESSION.get(url, timeout=_SESSION_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
        except Exception:
            results = None

        if results:
            elevation = results[0].get("elevation")
            if elevation is not None:
                return elevation

        if attempt < _RETRY_ATTEMPTS - 1:
            time.sleep(_RETRY_SLEEP * (attempt + 1))

    return None


def get_elevations(coords: List[Tuple[float, float]]) -> List[Optional[float]]:
    """
    Fetches elevation for a list of coordinates with cache and batch requests.
    
    Args:
        coords: List of coordinates (lat, lon)
    
    Returns:
        List of corresponding elevations (None if not found)
    """
    _load_cache()

    results: List[Optional[float]] = []
    missing: Dict[str, Tuple[float, float]] = {}

    with _CACHE_LOCK:
        for lat, lon in coords:
            key = _coord_key(lat, lon)
            if key in _CACHE:
                results.append(_CACHE[key])
            else:
                results.append(None)
                missing[key] = (lat, lon)

    if missing:
        for batch in _batched(missing.values(), _MAX_BATCH_SIZE):
            fetched = _fetch_batch(batch)
            with _CACHE_LOCK:
                for key, value in fetched.items():
                    if value is not None:
                        _CACHE[key] = value
            for idx, (lat, lon) in enumerate(coords):
                key = _coord_key(lat, lon)
                if key in fetched and results[idx] is None:
                    results[idx] = fetched[key]

        _save_cache()

    return results


def get_elevation(lat: float, lon: float) -> Optional[float]:
    """
    Fetches elevation of a point using cache and batch requests.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Elevation in meters or None if not found
    """
    returned = get_elevations([(lat, lon)])
    return returned[0] if returned else None


def horizontal_displacement_m(lat1, lon1, lat2, lon2):
    """
    Calculates horizontal displacement between two geographic points.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
    
    Returns:
        Tuple (x, y, dist_horizontal) in meters
        - x: east displacement (positive for east)
        - y: north displacement (positive for north)
        - dist_horizontal: horizontal distance (sqrt(x^2 + y^2))
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    λ1, λ2 = math.radians(lon1), math.radians(lon2)
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    x = dλ * math.cos((φ1 + φ2) / 2) * EARTH_RADIUS_M
    y = dφ * EARTH_RADIUS_M
    dist = math.hypot(x, y)
    return x, y, dist

def street_steepness(lat1, lon1, h1, lat2, lon2, h2):
    """
    Calculates the steepness of a street between two points.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        h1: Elevation of first point (meters)
        lat2: Latitude of second point
        lon2: Longitude of second point
        h2: Elevation of second point (meters)
    
    Returns:
        Dictionary with:
        - dh_m: elevation difference (h2 - h1) in meters
        - grade: ratio dh / dist_horizontal (None if indeterminate)
        - inclination_deg: vertical inclination angle in degrees
        - inclination_percent: inclination in percentage (None if indeterminate)
    """
    x, y, dist_h = horizontal_displacement_m(lat1, lon1, lat2, lon2)
    dh = h2 - h1
    dist_3d = math.hypot(dist_h, dh)

    inclination_rad = math.atan2(dh, dist_h)
    inclination_deg = math.fabs(math.degrees(inclination_rad))

    if dist_h == 0:
        grade = None
        inclination_percent = None
    else:
        grade = dh / dist_h
        inclination_percent = 100.0 * grade

    return {
        "dh_m": dh,
        "grade": grade,
        "inclination_deg": inclination_deg,
        "inclination_percent": inclination_percent
    }