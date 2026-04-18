"""
Visualize RAM pricing optimization results.

Usage:
    python visualize_results.py \\
        --results_dir data/results/UFL_Orlando_Thu \\
        --output data/results/UFL_Orlando_Thu/summary_plot.png

Produces a 3x2 panel figure:
    A) Operating Profit   D) Load Factor
    B) RASM               E) % Repositioning Flights
    C) UAM Passengers     F) Flights per Aircraft per Day
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from pathlib import Path


def _discover_runs(results_dir: Path):
    """Scan results dir for {fleet}_{casm_x10}.csv files. Returns sorted (fleet, casm) pairs."""
    pattern = re.compile(r"^(\d+)_(\d+)\.csv$")
    runs = []
    for f in results_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            runs.append((int(m.group(1)), int(m.group(2))))
    return sorted(runs)


def build_result_table(results_dir: Path) -> pd.DataFrame:
    runs = _discover_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    rows = []
    for fleet, casm_x10 in runs:
        opex = casm_x10 / 10.0
        rev_path  = results_dir / f"{fleet}_{casm_x10}.csv"
        repo_path = results_dir / f"repo_{fleet}_{casm_x10}.csv"

        try:
            rev  = pd.read_csv(rev_path)
            repo = pd.read_csv(repo_path)
        except FileNotFoundError:
            continue

        # Repositioning flights (different origin/destination only)
        repo_actual = repo[repo["origin_vertiport_id"] != repo["destination_vertiport_id"]]
        num_repo_flights = repo_actual["Value"].sum() if "Value" in repo.columns else 0

        # Excess empty flights counted as repositioning
        rev["excess_num_flights"] = rev.apply(
            lambda row: np.floor(max(0, row["num_flights"] - row["uam_pax"] / 4)), axis=1
        )

        total_rev  = rev["total_revenue"].sum()
        rev["cost"] = rev.apply(
            lambda row: (row["distance"] * opex * 4 + 20) * row["num_flights"], axis=1
        )
        total_cost = repo["cost"].sum() + rev["cost"].sum()
        profit     = total_rev - total_cost

        uam_pax    = rev["uam_pax"].sum()
        total_flights     = rev["num_flights"].sum()
        total_repo_flights = rev["excess_num_flights"].sum() + num_repo_flights
        percentage_repo    = total_repo_flights / (total_flights + num_repo_flights) if (total_flights + num_repo_flights) > 0 else 0
        load_factor        = uam_pax / (total_flights * 4) if total_flights > 0 else 0

        total_aircraft_miles = (
            (rev["distance"] * rev["num_flights"] * 4).sum()
            + (repo["repositioning_distance"] * repo.get("Value", 0) * 4).sum()
        )
        casm = total_cost / total_aircraft_miles if total_aircraft_miles > 0 else 0
        rasm = total_rev  / total_aircraft_miles if total_aircraft_miles > 0 else 0

        rows.append({
            "fleet_size":                   fleet,
            "operating_cost":               round(opex, 1),
            "revenue":                      total_rev,
            "total_cost":                   total_cost,
            "profit":                       profit,
            "num_flights":                  total_flights,
            "uam_pax":                      uam_pax,
            "casm":                         casm,
            "rasm":                         rasm,
            "load_factor":                  load_factor,
            "percentage_repo":              percentage_repo * 100,
            "num_flight_per_aircraft_per_day": (total_flights + num_repo_flights) / fleet,
        })

    return pd.DataFrame(rows)


def plot(result: pd.DataFrame, output_path: Path | None = None):
    n_costs = result["operating_cost"].nunique()
    palette  = sns.color_palette("coolwarm_r", n_costs)

    fig, ax = plt.subplots(3, 2, figsize=(12, 9), dpi=120)

    plot_cfg = [
        (ax[0, 0], "profit",                        "Operating Profit\n(thousand $)", "A)"),
        (ax[0, 1], "load_factor",                   "Load Factor",                    "D)"),
        (ax[1, 0], "rasm",                          "RASM ($)",                       "B)"),
        (ax[1, 1], "percentage_repo",               "% Repo. Flights",                "E)"),
        (ax[2, 0], "uam_pax",                       "# UAM Passengers",               "C)"),
        (ax[2, 1], "num_flight_per_aircraft_per_day","# Flights per Aircraft",         "F)"),
    ]

    for i, (axis, col, ylabel, label) in enumerate(plot_cfg):
        is_last = (i == len(plot_cfg) - 1)
        sns.lineplot(
            data=result, x="fleet_size", y=col,
            hue="operating_cost", marker="o",
            palette=palette, ax=axis,
            legend=is_last, markersize=6, linewidth=0.8,
        )
        show_xlabel = axis in (ax[2, 0], ax[2, 1])
        axis.set_xlabel("Fleet Size" if show_xlabel else "", fontsize=20)
        axis.set_xticklabels([] if not show_xlabel else axis.get_xticklabels())
        axis.set_ylabel(ylabel, fontsize=20)
        axis.tick_params(axis="both", which="major", labelsize=14)
        axis.text(0.02, 0.98, label, transform=axis.transAxes,
                  fontsize=18, va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1.5))

        # Profit: scale y-axis labels to thousands
        if col == "profit":
            ticks = axis.get_yticks()
            axis.set_yticklabels([f"{int(x / 1000)}" for x in ticks])

    # Shared grid and x-axis settings
    fleet_min = int(result["fleet_size"].min())
    fleet_max = int(result["fleet_size"].max())
    for row_axes in ax:
        for axis in row_axes:
            axis.set_xlim(fleet_min, fleet_max)
            axis.xaxis.set_major_locator(MultipleLocator(max(1, (fleet_max - fleet_min) // 6)))
            axis.xaxis.set_minor_locator(MultipleLocator(5))
            axis.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
            axis.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)

    # Legend
    handles, labels = ax[2, 1].get_legend_handles_labels()
    ax[2, 1].get_legend().remove()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.85, 0.5),
        title="CASM\n($/seat-mile)", title_fontsize=16,
        fontsize=14, borderaxespad=0.,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[visualize] Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize RAM pricing results")
    parser.add_argument("--results_dir", required=True,
                        help="Folder containing result CSVs (e.g. data/results/UFL_Orlando_Thu)")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path instead of showing it (e.g. data/results/UFL_Orlando_Thu/summary.png)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    result = build_result_table(results_dir)

    print(f"[visualize] Loaded {len(result)} runs - "
          f"fleet {result['fleet_size'].min()}-{result['fleet_size'].max()}, "
          f"CASM {result['operating_cost'].min()}-{result['operating_cost'].max()}")

    output_path = Path(args.output) if args.output else None
    plot(result, output_path)


if __name__ == "__main__":
    main()
