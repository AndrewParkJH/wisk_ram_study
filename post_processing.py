"""
Post-processing and visualization for RAM pricing optimization results.

Usage:
    python post_processing.py \\
        --results_dir  data/results/case2_costScalingFrom4Seater \\
        --city_pair    Chicago_UIUC_Thu \\
        --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UIUC.json

    # optional:
    python post_processing.py ... --seats 4
    python post_processing.py ... --optimizer_config configs/optimizer.json

Produces one 3x2 panel figure per seat-capacity subdirectory (or just --seats if given):
    A) Operating Profit   D) Load Factor
    B) Revenue            E) % Repositioning Flights
    C) UAM Passengers     F) Flights per Aircraft per Day

Output files:
    {results_dir}/{city_pair}/{num_seats}seats/summary_plot.png
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from pathlib import Path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_runs(seat_dir: Path):
    """Return sorted list of (fleet: int, pct: int) from f{fleet}_p{pct}.csv files."""
    pattern = re.compile(r"^f(\d+)_p(\d+)\.csv$")
    runs = []
    for f in seat_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            runs.append((int(m.group(1)), int(m.group(2))))
    return sorted(runs)


# ---------------------------------------------------------------------------
# Build result table for one seat-capacity directory
# ---------------------------------------------------------------------------

def build_result_table(
    seat_dir: Path,
    cost_cfg: dict,
    flight_time_matrix: np.ndarray,
    flight_distance_matrix: np.ndarray,
    num_seats: int,
) -> pd.DataFrame:
    """
    Load all runs in seat_dir and return a tidy DataFrame with one row per run.

    cost_cfg    : cfg["cost"] dict - contains normal_cost_per_seat_fh/fc
    pct tag     : integer from filename (e.g. 40, 100, 160) representing the
                  cost scale factor × 100 relative to the normal operating cost.
                  cost_per_fh = normal_cost_per_seat_fh * (pct/100) * num_seats
    """
    runs = _discover_runs(seat_dir)
    if not runs:
        raise FileNotFoundError(f"No f{{fleet}}_p{{pct}}.csv files found in {seat_dir}")

    fh_norm = cost_cfg["normal_cost_per_seat_fh"]
    fc_norm = cost_cfg["normal_cost_per_seat_fc"]

    rows = []
    for fleet, pct in runs:
        pct_scale   = pct / 100.0
        cost_per_fh = fh_norm * pct_scale * num_seats
        cost_per_fc = fc_norm * pct_scale * num_seats

        tag       = f"f{fleet}_p{pct}"
        rev_path  = seat_dir / f"{tag}.csv"
        repo_path = seat_dir / f"repo_{tag}.csv"

        try:
            rev  = pd.read_csv(rev_path)
            repo = pd.read_csv(repo_path)
        except FileNotFoundError:
            continue

        # --- Revenue flight costs ---
        rev_oi = rev["origin_vertiport_id"].astype(int).values
        rev_di = rev["destination_vertiport_id"].astype(int).values
        rev_ft_hrs = np.array([flight_time_matrix[o, d] for o, d in zip(rev_oi, rev_di)]) / 60.0
        rev_dist   = np.array([flight_distance_matrix[o, d] for o, d in zip(rev_oi, rev_di)])

        rev["cost"]     = (cost_per_fh * rev_ft_hrs + cost_per_fc) * rev["num_flights"].values
        rev["asm"]      = rev["num_flights"].values * rev_dist * num_seats
        asm_revenue_only = rev["asm"].sum()

        # --- Repositioning metrics ---
        repo_oi = repo["origin_vertiport_id"].astype(int).values
        repo_di = repo["destination_vertiport_id"].astype(int).values
        repo["flight_time_hrs"]        = [flight_time_matrix[o, d] / 60.0 for o, d in zip(repo_oi, repo_di)]
        repo["repositioning_distance"] = [flight_distance_matrix[o, d] for o, d in zip(repo_oi, repo_di)]
        repo["asm"]                    = repo["Value"].values * repo["repositioning_distance"].values * num_seats

        repo_actual      = repo[repo_oi != repo_di]
        num_repo_flights = repo_actual["Value"].sum()
        repo_cost        = ((cost_per_fh * repo_actual["flight_time_hrs"] + cost_per_fc)
                            * repo_actual["Value"]).sum()
        repo_asm         = repo["asm"].sum()

        asm_all_flights = asm_revenue_only + repo_asm

        # --- Aggregates ---
        total_rev   = rev["total_revenue"].sum()
        total_cost  = rev["cost"].sum() + repo_cost
        profit      = total_rev - total_cost

        uam_pax            = rev["uam_pax"].sum()
        total_rev_flights  = rev["num_flights"].sum()
        total_flights      = total_rev_flights + num_repo_flights
        load_factor        = uam_pax / (total_rev_flights * num_seats) if total_rev_flights > 0 else 0
        pct_repo           = num_repo_flights / total_flights * 100 if total_flights > 0 else 0
        flights_per_ac     = total_flights / fleet if fleet > 0 else 0

        casm_revenue_only = total_cost / asm_revenue_only if asm_revenue_only > 0 else 0
        casm_all_flights  = total_cost / asm_all_flights  if asm_all_flights  > 0 else 0
        rasm               = total_rev  / asm_revenue_only if asm_revenue_only > 0 else 0
        avg_stage_length   = (asm_revenue_only / (total_rev_flights * num_seats)
                              if total_rev_flights > 0 else 0)
        repo_cost_share    = repo_cost / total_cost if total_cost > 0 else 0
        repo_asm_share     = repo_asm  / asm_all_flights if asm_all_flights > 0 else 0

        total_demand    = rev["num_pax"].sum()
        demand_capture  = uam_pax / total_demand if total_demand > 0 else 0

        rows.append({
            "fleet_size":                      fleet,
            "cost_pct":                        pct,
            "cost_scale":                      pct_scale,
            "cost_per_fh":                     round(cost_per_fh, 1),
            "cost_per_fc":                     round(cost_per_fc, 1),
            "revenue":                         total_rev,
            "total_cost":                      total_cost,
            "profit":                          profit,
            "num_flights":                     total_rev_flights,
            "uam_pax":                         uam_pax,
            "load_factor":                     load_factor,
            "percentage_repo":                 pct_repo,
            "num_flight_per_aircraft_per_day": flights_per_ac,
            "asm_revenue_only":                asm_revenue_only,
            "asm_all_flights":                 asm_all_flights,
            "casm_rev":                        casm_revenue_only,
            "casm_all":                        casm_all_flights,
            "rasm":                            rasm,
            "avg_stage_length_mi":             avg_stage_length,
            "repo_cost_share":                 repo_cost_share,
            "repo_asm_share":                  repo_asm_share,
            "demand_capture":                  demand_capture,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(result: pd.DataFrame, num_seats: int, output_path: Path | None = None):
    scales   = sorted(result["cost_scale"].unique())
    n_scales = len(scales)
    palette  = sns.color_palette("coolwarm_r", n_scales)

    fig, ax = plt.subplots(3, 2, figsize=(12, 9), dpi=120)
    fig.suptitle(f"{num_seats}-seat vehicle", fontsize=16, y=1.01)

    plot_cfg = [
        (ax[0, 0], "profit",                          "Operating Profit\n(thousand $)", "A)"),
        (ax[0, 1], "load_factor",                     "Load Factor",                    "D)"),
        (ax[1, 0], "revenue",                         "Revenue ($)",                    "B)"),
        (ax[1, 1], "percentage_repo",                 "% Repo. Flights",                "E)"),
        (ax[2, 0], "uam_pax",                         "# UAM Passengers",               "C)"),
        (ax[2, 1], "num_flight_per_aircraft_per_day", "# Flights per Aircraft",         "F)"),
    ]

    for i, (axis, col, ylabel, label) in enumerate(plot_cfg):
        is_last = (i == len(plot_cfg) - 1)
        sns.lineplot(
            data=result, x="fleet_size", y=col,
            hue="cost_scale", marker="o",
            palette=palette, ax=axis,
            legend=is_last, markersize=6, linewidth=0.8,
        )
        show_xlabel = axis in (ax[2, 0], ax[2, 1])
        axis.set_xlabel("Fleet Size" if show_xlabel else "", fontsize=20)
        if not show_xlabel:
            axis.set_xticklabels([])
        axis.set_ylabel(ylabel, fontsize=20)
        axis.tick_params(axis="both", which="major", labelsize=14)
        axis.text(0.02, 0.98, label, transform=axis.transAxes,
                  fontsize=18, va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1.5))

        if col == "profit":
            ticks = axis.get_yticks()
            axis.set_yticklabels([f"{int(x / 1000)}" for x in ticks])

    fleet_min = int(result["fleet_size"].min())
    fleet_max = int(result["fleet_size"].max())
    for row_axes in ax:
        for axis in row_axes:
            axis.set_xlim(fleet_min, fleet_max)
            axis.xaxis.set_major_locator(MultipleLocator(max(1, (fleet_max - fleet_min) // 6)))
            axis.xaxis.set_minor_locator(MultipleLocator(5))
            axis.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
            axis.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)

    # Legend: cost scale values as ×0.40 … ×1.60
    handles, labels = ax[2, 1].get_legend_handles_labels()
    ax[2, 1].get_legend().remove()
    scale_labels = [f"\u00d7{float(l):.2f}" for l in labels]
    fig.legend(
        handles, scale_labels,
        loc="center left", bbox_to_anchor=(0.85, 0.5),
        title="Cost Scale\n(×normal)", title_fontsize=14,
        fontsize=12, borderaxespad=0.,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved → {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process and visualize RAM pricing results")
    parser.add_argument("--results_dir",      required=True,
                        help="Base results directory (e.g. data/results/case2_costScalingFrom4Seater)")
    parser.add_argument("--city_pair",        required=True,
                        help="City-pair subdirectory to process (e.g. Chicago_UIUC_Thu)")
    parser.add_argument("--vertiport_config", required=True,
                        help="Path to vertiport config JSON")
    parser.add_argument("--optimizer_config", default="configs/optimizer.json",
                        help="Shared optimizer config JSON (default: configs/optimizer.json)")
    parser.add_argument("--seats", type=int, default=None,
                        help="Only process this seat count (default: all)")
    args = parser.parse_args()

    root = Path(__file__).parent

    with open(root / args.optimizer_config) as f:
        cfg = json.load(f)

    with open(root / args.vertiport_config) as f:
        vp_cfg = json.load(f)

    ft_matrix = np.array(vp_cfg["links"]["flight_time_matrix"],     dtype=float)
    fd_matrix = np.array(vp_cfg["links"]["flight_distance_matrix"], dtype=float)

    cost_cfg = cfg["cost"]
    base_out = root / args.results_dir / args.city_pair

    seat_list = cfg["num_seats"]
    if args.seats is not None:
        seat_list = [s for s in seat_list if s == args.seats]
        if not seat_list:
            raise ValueError(f"--seats {args.seats} not found in optimizer config num_seats list")

    plots_dir = root / args.results_dir / "summary_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for num_seats in seat_list:
        seat_dir = base_out / f"{num_seats}seats"
        if not seat_dir.exists():
            print(f"[post_processing] Skipping {seat_dir} - directory not found")
            continue

        print(f"[post_processing] Processing {num_seats}-seat results in {seat_dir} ...")
        result = build_result_table(seat_dir, cost_cfg, ft_matrix, fd_matrix, num_seats)
        print(
            f"[post_processing]   {len(result)} runs - "
            f"fleet {result['fleet_size'].min()}-{result['fleet_size'].max()}, "
            f"cost scale {sorted(result['cost_scale'].unique())}"
        )

        output_path = plots_dir / f"{args.city_pair}_{num_seats}seats_summary.png"
        plot(result, num_seats, output_path)

    print("[post_processing] Done.")


if __name__ == "__main__":
    main()
