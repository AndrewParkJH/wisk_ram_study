"""
Builds all static and dynamic inputs needed by StarNetwork and PricingOptimizer.

Reads:
  - vertiport config JSON  (vertiports + links)
  - demand dirs            (*_trips.csv files from run_pipeline.py output)
                           Both directions are loaded automatically:
                             {base_demand_dir}/{o_city}_{d_city}_{day}/
                             {base_demand_dir}/{d_city}_{o_city}_{day}/

Returns a NetworkInputs dataclass containing every matrix the optimizer needs,
derived entirely from config and demand data — no hardcoding.

city_pair format
----------------
  "{o_city}_{d_city}_{day}"   e.g. "UFL_Orlando_Thu"
  Day must be the last token and one of: Thu, Sat
  o_city and d_city are resolved against the city names in the vertiport config.

Config links format
-------------------
"links": {
    "nodes": ["HUB", "SPOKE_A", "SPOKE_B", ...],   <- defines matrix row/column order
    "flight_time_matrix":     [[0,36,...], ...],     <- minutes, N x N
    "flight_distance_matrix": [[0,90,...], ...]      <- miles,   N x N
}
"""

import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "external" / "replica_data_analytics"))

DEFAULT_DEMAND_BASE = ROOT / "data" / "processed_data" / "demand"
VALID_DAYS = {"Thu", "Sat"}


@dataclass
class NetworkInputs:
    vertiport_names: List[str]            # ordered per links.nodes
    flight_distance_matrix: np.ndarray    # (N x N) miles   — from config links
    flight_time_matrix: np.ndarray        # (N x N) minutes — from config links
    energy_consumption_matrix: np.ndarray # (N x N) zeros   — unused by optimizer
    demand_df: pd.DataFrame               # hourly OD counts for StarNetwork.load_demand()
    driving_travel_time: np.ndarray       # (N x N x 24) minutes — from demand CSVs
    driving_cost: np.ndarray              # (N x N x 24) USD     — from demand CSVs
    first_mile_time: np.ndarray           # (N x 24)     minutes — from demand CSVs
    last_mile_time: np.ndarray            # (N x 24)     minutes — from demand CSVs
    first_or_last_cost: np.ndarray        # (N x N x 24) USD     — from demand CSVs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_od_stem(stem: str, vp_set: set):
    """
    Split a filename stem like 'NO_UFL' into (origin_id, destination_id)
    by trying every underscore position until both parts are valid vertiport IDs.
    """
    parts = stem.split("_")
    for i in range(1, len(parts)):
        o = "_".join(parts[:i])
        d = "_".join(parts[i:])
        if o in vp_set and d in vp_set:
            return o, d
    return None, None


def _parse_city_pair(city_pair: str, city_names: set):
    """
    Given a city_pair string like "UFL_Orlando_Thu" return (o_city, d_city, day).

    The day is the last token if it is in VALID_DAYS, otherwise raises ValueError.
    The remaining prefix is split into two known city names by trying every
    underscore position — the same logic as _split_od_stem.
    """
    parts = city_pair.split("_")
    if parts[-1] not in VALID_DAYS:
        raise ValueError(
            f"city_pair '{city_pair}' must end with one of {VALID_DAYS}. "
            f"Got '{parts[-1]}'."
        )
    day = parts[-1]
    base = "_".join(parts[:-1])          # e.g. "UFL_Orlando"

    base_parts = base.split("_")
    for i in range(1, len(base_parts)):
        o = "_".join(base_parts[:i])
        d = "_".join(base_parts[i:])
        if o in city_names and d in city_names:
            return o, d, day

    raise ValueError(
        f"Cannot split '{base}' into two known city names.\n"
        f"  Known cities from vertiport config: {sorted(city_names)}\n"
        f"  Tip: city_pair must be '{{o_city}}_{{d_city}}_{{day}}' where both "
        f"city names match the 'city' field in the vertiport config."
    )


def _verify(vp_set: set, dist_matrix: np.ndarray, vp_idx: dict, network_files: list):
    """
    Validate that every demand file pair has matching nodes and a non-zero matrix entry.
    Errors are fatal.
    """
    errors = []

    for f, o_vp, d_vp in network_files:
        if o_vp not in vp_set:
            errors.append(f"  Origin '{o_vp}' in {f.name} not found in links.nodes")
        if d_vp not in vp_set:
            errors.append(f"  Destination '{d_vp}' in {f.name} not found in links.nodes")
        if o_vp in vp_idx and d_vp in vp_idx:
            oi, di = vp_idx[o_vp], vp_idx[d_vp]
            if dist_matrix[oi, di] == 0:
                errors.append(
                    f"  links.flight_distance_matrix[{o_vp},{d_vp}] is 0 "
                    f"— add distance for {f.name}"
                )

    if errors:
        print("[build_network] ERROR — fix config before running:")
        for e in errors:
            print(e)
        raise ValueError("Network verification failed.")

    print(f"[build_network] Verification passed: {len(network_files)} OD pairs matched.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_network_inputs(
    vertiport_config_path: str,
    city_pair: str,
    start_hour: int,
    base_demand_dir: str = None,
) -> NetworkInputs:
    """
    Parameters
    ----------
    vertiport_config_path : path to the vertiport JSON config
    city_pair             : "{o_city}_{d_city}_{day}" e.g. "UFL_Orlando_Thu"
    start_hour            : ignore demand arriving before this hour
    base_demand_dir       : root folder containing per-study demand subdirs;
                            defaults to <repo>/data/processed_data/demand
    """
    with open(vertiport_config_path) as f:
        config = json.load(f)

    base_demand_dir = Path(base_demand_dir) if base_demand_dir else DEFAULT_DEMAND_BASE

    # --- Vertiport ordering from links.nodes ---
    links = config.get("links")
    if not links:
        raise ValueError(f"No 'links' section in {vertiport_config_path}.")

    vertiport_names = links["nodes"]
    vp_set = set(vertiport_names)
    vp_idx = {name: i for i, name in enumerate(vertiport_names)}
    N = len(vertiport_names)

    # --- Static matrices directly from config ---
    flight_distance_matrix    = np.array(links["flight_distance_matrix"], dtype=float)
    flight_time_matrix        = np.array(links["flight_time_matrix"],     dtype=float)
    energy_consumption_matrix = np.zeros((N, N))

    if flight_distance_matrix.shape != (N, N):
        raise ValueError(f"flight_distance_matrix must be {N}x{N}, got {flight_distance_matrix.shape}")
    if flight_time_matrix.shape != (N, N):
        raise ValueError(f"flight_time_matrix must be {N}x{N}, got {flight_time_matrix.shape}")

    # --- Resolve both demand directories from city_pair ---
    city_names = {vp["city"] for vp in config["vertiports"]}
    o_city, d_city, day = _parse_city_pair(city_pair, city_names)

    forward_dir = base_demand_dir / f"{o_city}_{d_city}_{day}"
    reverse_dir = base_demand_dir / f"{d_city}_{o_city}_{day}"

    demand_dirs = []
    for d in (forward_dir, reverse_dir):
        if d.exists():
            demand_dirs.append(d)
        else:
            print(f"[build_network] WARNING: demand directory not found, skipping: {d}")

    if not demand_dirs:
        raise FileNotFoundError(
            f"Neither demand directory found for city_pair '{city_pair}':\n"
            f"  {forward_dir}\n  {reverse_dir}"
        )

    print(f"[build_network] Loading demand from {len(demand_dirs)} direction(s): "
          + ", ".join(d.name for d in demand_dirs))

    # --- Discover demand files across both dirs ---
    all_trip_files = []
    for d in demand_dirs:
        all_trip_files.extend(sorted(d.glob("*_trips.csv")))

    network_files = []
    for f in all_trip_files:
        stem = f.stem.replace("_trips", "")
        o_vp, d_vp = _split_od_stem(stem, vp_set)
        if o_vp is not None:
            network_files.append((f, o_vp, d_vp))

    if not network_files:
        raise ValueError(
            f"No demand files in {[d.name for d in demand_dirs]} match "
            f"vertiports in {vertiport_config_path}.\n"
            f"  links.nodes: {vertiport_names}\n"
            f"  Files found: {[f.name for f in all_trip_files]}"
        )

    # --- Verify before doing any work ---
    _verify(vp_set, flight_distance_matrix, vp_idx, network_files)

    # --- Dynamic matrices from demand CSVs ---
    driving_travel_time = np.zeros((N, N, 24))
    driving_cost        = np.zeros((N, N, 24))
    first_mile_time     = np.zeros((N, 24))
    last_mile_time      = np.zeros((N, 24))
    first_or_last_cost  = np.zeros((N, N, 24))
    demand_dfs = []

    for f, o_vp, d_vp in network_files:
        df = pd.read_csv(f, low_memory=False)
        oi, di = vp_idx[o_vp], vp_idx[d_vp]

        # Filter by service start hour (based on when passenger arrives at origin vertiport)
        origin_arrival_hour = pd.to_datetime(
            df["arrival_time_at_origin_vertiport"], errors="coerce"
        ).dt.hour
        df = df[origin_arrival_hour >= start_hour]

        # Hourly demand keyed by origin_vertiport_arrival_hour.
        # Multiply by 365: ScheduleGenerator internally divides by 365 (treats input
        # as annual demand), so we scale up to recover daily ridership.
        hourly = (
            df.groupby("origin_vertiport_arrival_hour")
              .size()
              .reset_index(name="total_trips")
              .rename(columns={"origin_vertiport_arrival_hour": "hour"})
        )
        hourly["total_trips"] *= 365
        hourly["od"] = f"{o_vp}_{d_vp}"
        demand_dfs.append(hourly)

        # Per-hour cost aggregates (indexed by departure hour)
        agg = df.groupby("depart_hour").agg(
            driving_ivtt = ("Driving_IVTT_min", "mean"),
            driving_fare = ("Driving_Fare_USD",  "mean"),
            fm_duration  = ("FM_duration_min",   "mean"),
            fm_fare      = ("FM_fare_USD",        "mean"),
            lm_fare      = ("LM_fare_USD",        "mean"),
        ).reset_index()

        for _, row in agg.iterrows():
            h = int(row["depart_hour"])
            driving_travel_time[oi, di, h] = row["driving_ivtt"]
            driving_cost[oi, di, h]        = row["driving_fare"]
            first_mile_time[oi, h]         = row["fm_duration"]
            first_or_last_cost[oi, di, h]  = row["fm_fare"] + row["lm_fare"]

        # Last-mile indexed by destination vertiport + arrival hour at destination
        lm_agg = df.groupby("last_hour").agg(
            lm_duration = ("LM_duration_min", "mean")
        ).reset_index()

        for _, row in lm_agg.iterrows():
            h = int(row["last_hour"])
            last_mile_time[di, h] = row["lm_duration"]

    demand_df = pd.concat(demand_dfs, ignore_index=True)

    print(f"[build_network] Built inputs for {N} vertiports: {vertiport_names}")
    print(f"[build_network] Loaded {len(network_files)} OD demand files.")

    return NetworkInputs(
        vertiport_names           = vertiport_names,
        flight_distance_matrix    = flight_distance_matrix,
        flight_time_matrix        = flight_time_matrix,
        energy_consumption_matrix = energy_consumption_matrix,
        demand_df                 = demand_df,
        driving_travel_time       = driving_travel_time,
        driving_cost              = driving_cost,
        first_mile_time           = first_mile_time,
        last_mile_time            = last_mile_time,
        first_or_last_cost        = first_or_last_cost,
    )
