"""
Entry point for the RAM pricing optimization.

Usage:
    python run_pricing.py \\
        --city_pair        UFL_Orlando_Thu \\
        --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json

    # override shared optimizer settings (optional):
    python run_pricing.py ... --optimizer_config configs/optimizer.json

Output structure:
    data/results/{city_pair}/{num_seats}seats/
        f{fleet}_p{cost_pct}.csv        <- optimisation results
        repo_f{fleet}_p{cost_pct}.csv   <- repositioning flights

    cost_pct = 0..100  (percentage along the [min, max] cost range defined per seat config)
    e.g.  f10_p0.csv  = fleet 10, minimum cost point
          f10_p50.csv = fleet 10, midpoint cost
          f10_p100.csv= fleet 10, maximum cost point

Vertiport config supplies (per vertiport, indexed as links.nodes):
    fato_capacity        — list passed to optimizer
    middle_mile ovtt_min — used as uam_transition_time

Shared optimizer config (configs/optimizer.json) supplies:
    optimizer settings, fleet sweep, seat configs, output_base_dir
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

sys.path.insert(0, str(ROOT / "external" / "replica_data_analytics"))
sys.path.insert(0, str(ROOT / "external" / "uam_system_model"))

from uam_system_model.StarNetworkJFK import StarNetwork
from uam_system_model.PricingFH import PricingOptimizer
from build_network_inputs import build_network_inputs


def _load_vertiport_params(vertiport_config_path: str):
    """
    Extract optimizer-level parameters from the vertiport config:
      - uam_transition_time : middle_mile ovtt_min (minutes)
      - fato_capacity_list  : list of per-vertiport fato_capacity, ordered as links.nodes
    """
    with open(vertiport_config_path) as f:
        vp_cfg = json.load(f)

    # uam_transition_time = middle_mile OVTT
    segments = vp_cfg["assumptions"]["multimodal_segments"]["option1_ram"]
    mm_ovtt = next(s["ovtt_min"] for s in segments if s["segment"] == "middle_mile")

    # fato_capacity list ordered by links.nodes
    node_order = vp_cfg["links"]["nodes"]
    vp_map = {v["vertiport_id"]: v for v in vp_cfg["vertiports"]}
    fato_list = [vp_map[vp_id]["fato_capacity"] for vp_id in node_order]

    return mm_ovtt, fato_list

def main():
    parser = argparse.ArgumentParser(description="RAM pricing optimisation")
    parser.add_argument("--city_pair",        required=True, 
                        default="UFL_Orlando_Thu",
                        help="'{o_city}_{d_city}_{day}' e.g. UFL_Orlando_Thu — "
                             "both travel directions are loaded automatically")
    parser.add_argument("--vertiport_config", required=True, 
                        default="external/replica_data_analytics/config/vertiport_configuration/UFL.json",
                        help="Path to vertiport config JSON (e.g. external/.../UFL.json)")
    parser.add_argument("--optimizer_config", default="configs/optimizer.json",
                        help="Shared optimizer config JSON (default: configs/optimizer.json)")
    args = parser.parse_args()

    with open(ROOT / args.optimizer_config) as f:
        cfg = json.load(f)

    opt = cfg["optimizer"]

    # ── Extract per-vertiport params from vertiport config ─────────────────────
    vp_config_path = str(ROOT / args.vertiport_config)
    uam_transition_time, fato_capacity = _load_vertiport_params(vp_config_path)

    print(f"[run_pricing] uam_transition_time={uam_transition_time} min  "
          f"fato_capacity={fato_capacity}")

    # ── Build network inputs once (shared across all sweep iterations) ─────────
    inputs = build_network_inputs(
        vertiport_config_path = vp_config_path,
        city_pair             = args.city_pair,
        start_hour            = opt["start_hour"],
    )

    network = StarNetwork(
        vertiport_names           = inputs.vertiport_names,
        flight_distance_matrix    = inputs.flight_distance_matrix,
        flight_time_matrix        = inputs.flight_time_matrix,
        energy_consumption_matrix = inputs.energy_consumption_matrix,
    )
    network.load_demand(inputs.demand_df)
    optimizer = PricingOptimizer(StarNetwork=network)

    # ── Sweep ranges ───────────────────────────────────────────────────────────
    fleet_sizes = np.arange(
        cfg["fleet_sweep"]["min"],
        cfg["fleet_sweep"]["max"] + 1,
        cfg["fleet_sweep"]["step"],
    )
    cost_cfg = cfg["cost"]
    fh_norm  = cost_cfg["normal_cost_per_seat_fh"]
    fc_norm  = cost_cfg["normal_cost_per_seat_fc"]

    if cost_cfg["use_cost_multiplier"]:

        fh_min   = cost_cfg["cost_multiplier_fh"]["min"]
        fh_max   = cost_cfg["cost_multiplier_fh"]["max"]
        n_steps  = int(cost_cfg["cost_step"])   # number of evenly-spaced points
        pct_values = np.linspace(fh_min, fh_max, n_steps)

    else:
        pct_values = np.array([1.0])   # 1.0 multiplier → run at normal cost
                    

    base_out = ROOT / cfg["output_base_dir"] / args.city_pair

    # ── Main sweep: seat_config × fleet × cost_percentage ─────────────────────
    for num_seats in cfg["num_seats"]:
        out_dir = base_out / f"{num_seats}seats"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build a matching fc sweep (separate range or mirror fh)
        if cost_cfg["use_cost_multiplier"] and not cost_cfg["use_same_multiplier_for_fh_and_fc"]:
            fc_min   = cost_cfg["cost_multiplier_fc"]["min"]
            fc_max   = cost_cfg["cost_multiplier_fc"]["max"]
            fc_pct_values = np.linspace(fc_min, fc_max, int(cost_cfg["cost_step"]))
        else:
            fc_pct_values = pct_values  # same sweep for both

        for fleet in fleet_sizes:
            for pct_fh, pct_fc in zip(pct_values, fc_pct_values):
                cost_per_fh = (fh_norm * pct_fh)*num_seats
                cost_per_fc = (fc_norm * pct_fc)*num_seats
                pct_int     = round(pct_fh * 100)
                tag         = f"f{int(fleet)}_p{pct_int}"

                print(
                    f"[run_pricing] seats={num_seats}  fleet={int(fleet)}"
                    f"  fh={cost_per_fh:.0f} (×{pct_fh:.2f})"
                    f"  fc={cost_per_fc:.0f} (×{pct_fc:.2f})"
                )

                results, repo_flights = optimizer.optimize(
                    time_resolution       = opt["time_resolution"],
                    num_vehicles          = int(fleet),
                    uber_travel_time      = inputs.driving_travel_time,
                    uber_fare             = inputs.driving_cost,
                    first_mile_time       = inputs.first_mile_time,
                    last_mile_time        = inputs.last_mile_time,
                    first_and_last_cost   = inputs.first_or_last_cost,
                    uam_flight_time       = inputs.flight_time_matrix,
                    uam_distance_matrix   = inputs.flight_distance_matrix,
                    optimality_gap        = opt["optimality_gap"],
                    value_of_time         = opt["value_of_time"],
                    time_limit            = opt["time_limit"],
                    uam_transition_time   = uam_transition_time,
                    utility_type          = opt["utility_type"],
                    cost_per_fh           = cost_per_fh,
                    cost_per_fc           = cost_per_fc,
                    fato_capacity         = fato_capacity,
                    num_seats             = num_seats,
                    verbose               = False,
                )

                results.to_csv(out_dir / f"{tag}.csv",       index=False)
                repo_flights.to_csv(out_dir / f"repo_{tag}.csv", index=False)

    print(f"[run_pricing] Done. Results written to {base_out}")


if __name__ == "__main__":
    main()
