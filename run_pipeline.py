"""
Entry point for the RAM demand pipeline.

Usage:
    python run_pipeline.py \\
        --state Illinois \\
        --o_city Chicago \\
        --d_city UIUC \\
        --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UIUC.json

Data layout (all paths relative to this file):
    data/replica_data/<state>/<o_city>_<d_city>_<day>.csv   <- raw input (day: Thu or Sat)
    data/processed_data/od_cost_matrix/<state>/             <- generated OD matrices (cached)
    data/processed_data/demand/<o_city>_<d_city>_<day>/     <- final trip output
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

# Make the submodule importable
sys.path.insert(0, str(ROOT / "external" / "replica_data_analytics"))

from generate_regional_od_cost_lookup_tables import (
    create_hourly_od_lookup_tables,
    correct_hourly_od_matrices,
    save_trip_time_lookup_table,
    create_average_distance_od_matrix,
)
from generate_multimodal_trip_data import compute_ram_trip_statistics


RAW_DATA_DIR = ROOT / "data" / "replica_data"
PROCESSED_DATA_DIR = ROOT / "data" / "processed_data"


def _od_cost_exists(state: str, city: str, day: str) -> bool:
    tt_folder = PROCESSED_DATA_DIR / "od_cost_matrix" / state / f"{city}_{day}_hourly_od_lookup_tables"
    dist_file = PROCESSED_DATA_DIR / "od_cost_matrix" / state / "od_distance_matrix" / f"{city}_{day}_DIST_Matrix.csv"
    all_tt = all((tt_folder / f"{city}_{day}_TT_Matrix_{h:02d}.csv").exists() for h in range(24))
    return all_tt and dist_file.exists()


def _generate_od_cost(state: str, city: str, day: str) -> None:
    input_csv = RAW_DATA_DIR / state / f"{city}_{city}_{day}.csv"
    tt_output = PROCESSED_DATA_DIR / "od_cost_matrix" / state / f"{city}_{day}_hourly_od_lookup_tables"
    dist_output = PROCESSED_DATA_DIR / "od_cost_matrix" / state / "od_distance_matrix"

    print(f"[pipeline] Generating OD cost matrices for {city} ({day}) ...")
    lookup = create_hourly_od_lookup_tables(
        input_csv_path=str(input_csv),
        output_dir=str(tt_output),
    )
    lookup = correct_hourly_od_matrices(lookup)
    save_trip_time_lookup_table(lookup, output_folder=str(tt_output), file_name=f"{city}_{day}_TT_Matrix")
    create_average_distance_od_matrix(
        input_csv_path=str(input_csv),
        output_dir=str(dist_output),
        output_file_name=f"{city}_{day}_DIST_Matrix",
    )


def main():
    parser = argparse.ArgumentParser(description="RAM demand pipeline")
    parser.add_argument("--state", required=True, help="State name (e.g. Illinois)")
    parser.add_argument("--o_city", required=True, help="Origin city (e.g. Chicago)")
    parser.add_argument("--d_city", required=True, help="Destination city (e.g. UIUC)")
    parser.add_argument("--vertiport_config", required=True, help="Path to vertiport JSON config")
    parser.add_argument("--day", default="Thu", choices=["Thu", "Sat"], help="Day of week (default: Thu)")
    args = parser.parse_args()

    for city in [args.o_city, args.d_city]:
        if _od_cost_exists(args.state, city, args.day):
            print(f"[pipeline] OD cost matrices for {city} ({args.day}) already exist, skipping generation.")
        else:
            _generate_od_cost(args.state, city, args.day)

    output_dir = PROCESSED_DATA_DIR / "demand" / f"{args.o_city}_{args.d_city}_{args.day}"

    compute_ram_trip_statistics(
        veriport_spec_file=str(ROOT / args.vertiport_config),
        o_city=args.o_city,
        d_city=args.d_city,
        state=args.state,
        day=args.day,
        raw_data_dir=str(RAW_DATA_DIR),
        processed_data_dir=str(PROCESSED_DATA_DIR),
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
