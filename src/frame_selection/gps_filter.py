"""
gps_filter.py
──────────────
GPS proximity helpers for the frame selection cascade.

Loads a station database (stations.csv) and a per-video GPS track
(timestamp_sec, lat, lon).  For each frame index, interpolates the
truck's position from the track and checks whether it is within a
configurable radius of any known gas station.

GPS-flagged frames are sent directly to Stage 3 (full model) regardless
of Stage 2 outcome, because the truck is physically near a station.
"""

from __future__ import annotations
import csv
import math
from pathlib import Path


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_stations(csv_path: str | Path) -> list[dict]:
    """
    Load stations.csv.
    Expected columns: station_id, name, brand, lat, lon,
                      price_regular, price_midgrade, price_premium, price_diesel
    """
    stations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stations.append({
                "station_id":     row["station_id"],
                "name":           row["name"],
                "brand":          row["brand"],
                "lat":            float(row["lat"]),
                "lon":            float(row["lon"]),
                "price_regular":  row.get("price_regular",  "NA"),
                "price_midgrade": row.get("price_midgrade", "NA"),
                "price_premium":  row.get("price_premium",  "NA"),
                "price_diesel":   row.get("price_diesel",   "NA"),
            })
    return stations


def load_gps_track(csv_path: str | Path) -> list[dict]:
    """
    Load a per-video GPS track CSV.
    Expected columns: timestamp_sec, lat, lon
    Returns rows sorted by timestamp.
    """
    track = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            track.append({
                "timestamp_sec": float(row["timestamp_sec"]),
                "lat":           float(row["lat"]),
                "lon":           float(row["lon"]),
            })
    return sorted(track, key=lambda r: r["timestamp_sec"])


# ── Position interpolation ────────────────────────────────────────────────────

def get_frame_gps(
    frame_idx: int,
    fps:       float,
    track:     list[dict],
) -> tuple[float, float] | None:
    """
    Linearly interpolate the truck's (lat, lon) at a given frame index.
    Returns None if the track is empty.
    """
    if not track:
        return None

    t = frame_idx / fps

    if t <= track[0]["timestamp_sec"]:
        return track[0]["lat"], track[0]["lon"]
    if t >= track[-1]["timestamp_sec"]:
        return track[-1]["lat"], track[-1]["lon"]

    for i in range(len(track) - 1):
        t0 = track[i]["timestamp_sec"]
        t1 = track[i + 1]["timestamp_sec"]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            lat = track[i]["lat"] + alpha * (track[i + 1]["lat"] - track[i]["lat"])
            lon = track[i]["lon"] + alpha * (track[i + 1]["lon"] - track[i]["lon"])
            return lat, lon

    return None


# ── Proximity check ───────────────────────────────────────────────────────────

def nearest_station(
    lat:      float,
    lon:      float,
    stations: list[dict],
) -> tuple[dict, float]:
    """Return (nearest_station_dict, distance_m)."""
    best = min(stations, key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))
    return best, haversine(lat, lon, best["lat"], best["lon"])


def is_near_station(
    lat:      float,
    lon:      float,
    stations: list[dict],
    radius_m: float = 500.0,
) -> tuple[bool, dict | None, float]:
    """
    Check whether (lat, lon) is within radius_m of any known station.

    Returns
    -------
    (flagged, nearest_station_dict, distance_m)
    """
    if not stations:
        return False, None, float("inf")
    station, dist = nearest_station(lat, lon, stations)
    return dist <= radius_m, station, round(dist, 1)
