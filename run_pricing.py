"""
Entry point for the RAM pricing optimization.

Usage:
    python run_pricing.py \\
        --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json \\
        --demand_dir data/processed_data/demand/UFL_Orlando_Thu \\
        --output_dir data/results/UFL_Orlando_Thu

Optional sweep parameters:
    --fleet_min 10 --fleet_max 65 --fleet_step 5
    --casm_min 0.6 --casm_max 1.6 --casm_step 0.1
    --optimality_gap 0.05
    --time_limit 3600
"""

import sys
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

# Make both submodules importable
sys.path.insert(0, str(ROOT / "external" / "replica_data_analytics"))
sys.path.insert(0, str(ROOT / "external" / "uam_system_model"))

from uam_system_model.StarNetworkJFK import StarNetwork
from uam_system_model.Pricing import PricingOptimizer
from build_network_inputs import build_network_inputs


def main():
    parser = argparse.ArgumentParser(description="RAM pricing optimization")
    parser.add_argument("--vertiport_config", required=True,
                        help="Path to vertiport config JSON (e.g. external/replica_data_analytics/config/vertiport_configuration/UFL.json)")
    parser.add_argument("--demand_dir", required=True,
                        help="Directory containing *_trips.csv demand files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write optimization results")

    # Sweep parameters
    parser.add_argument("--fleet_min",  type=int,   default=10)
    parser.add_argument("--fleet_max",  type=int,   default=65)
    parser.add_argument("--fleet_step", type=int,   default=5)
    parser.add_argument("--casm_min",   type=float, default=0.6)
    parser.add_argument("--casm_max",   type=float, default=1.6)
    parser.add_argument("--casm_step",  type=float, default=0.1)

    # Optimizer parameters
    parser.add_argument("--optimality_gap", type=float, default=0.05)
    parser.add_argument("--time_limit",     type=int,   default=3600)
    parser.add_argument("--time_resolution",type=int,   default=30)
    parser.add_argument("--value_of_time",  type=float, default=32.63)
    parser.add_argument("--num_seats",      type=int,   default=4)
    parser.add_argument("--fato_capacity",  type=int,   default=10)
    parser.add_argument("--fixed_cost_per_flight", type=float, default=20.0)
    parser.add_argument("--uam_transition_time",   type=int,   default=10)
    parser.add_argument("--utility_type",   default="betas", choices=["betas", "vot"])
    parser.add_argument("--start_hour",     type=int, default=0,
                        help="Earliest hour (origin vertiport arrival) to include in demand (default: 0 = no filter)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build all inputs from config + demand files ---
    inputs = build_network_inputs(args.vertiport_config, args.demand_dir, args.start_hour)

    # --- Construct network ---
    network = StarNetwork(
        vertiport_names          = inputs.vertiport_names,
        flight_distance_matrix   = inputs.flight_distance_matrix,
        flight_time_matrix       = inputs.flight_time_matrix,
        energy_consumption_matrix= inputs.energy_consumption_matrix,
    )
    network.load_demand(inputs.demand_df)

    optimizer = PricingOptimizer(StarNetwork=network)

    # --- Sweep ---
    fleet_sizes  = np.arange(args.fleet_min,  args.fleet_max  + 1, args.fleet_step)
    casm_values  = np.arange(args.casm_min,   args.casm_max,        args.casm_step)

    for f in fleet_sizes:
        for c in casm_values:
            print(f"[run_pricing] fleet={f}  casm={c:.2f}")
            results, repo_flights = optimizer.optimize(
                time_resolution       = args.time_resolution,
                num_vehicles          = f,
                uber_travel_time      = inputs.driving_travel_time,
                uber_fare             = inputs.driving_cost,
                first_mile_time       = inputs.first_mile_time,
                last_mile_time        = inputs.last_mile_time,
                first_or_last_cost    = inputs.first_or_last_cost,
                uam_flight_time       = inputs.flight_time_matrix,
                uam_distance_matrix   = inputs.flight_distance_matrix,
                optimality_gap        = args.optimality_gap,
                value_of_time         = args.value_of_time,
                time_limit            = args.time_limit,
                uam_transition_time   = args.uam_transition_time,
                utility_type          = args.utility_type,
                opex_per_asm          = c,
                fato_capacity         = args.fato_capacity,
                num_seats             = args.num_seats,
                fixed_cost_per_flight = args.fixed_cost_per_flight,
                verbose               = False,
            )
            tag = f"{f}_{round(c * 10)}"
            results.to_csv(output_dir / f"{tag}.csv", index=False)
            repo_flights.to_csv(output_dir / f"repo_{tag}.csv", index=False)

    print(f"[run_pricing] Done. Results written to {output_dir}")


if __name__ == "__main__":
    main()
