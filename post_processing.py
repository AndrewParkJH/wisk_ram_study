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

        if asm_revenue_only == 0:
            continue

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
        profit      = (total_rev - total_cost)

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
# Plotting  (Plot_1: 3-row x 4-col market summary)
# ---------------------------------------------------------------------------

# Columns shown: A=Profit, B=Revenue, C=UAM Pax, F=Flights per Aircraft
PLOT_COLS = [
    ("profit",                          "A) Operating Profit (thousand $)"),
    ("revenue",                         "B) Revenue (thousand $)"),
    ("uam_pax",                         "C) UAM Passengers per Day"),
    ("num_flight_per_aircraft_per_day", "F) #Flights per Aircraft"),
]


def plot_market(
    results_by_seat: dict,          # {num_seats: DataFrame}
    city_pair: str,
    output_path: Path | None = None,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """
    3-row x 4-col summary figure for one market.
    Rows = seat configurations (sorted ascending).
    Cols = A) Profit, B) Revenue, C) UAM Pax, F) Flights per Aircraft.
    Lines are coloured by cost scaling factor.
    """
    seat_list = sorted(results_by_seat.keys())
    n_rows    = len(seat_list)      # 3
    n_cols    = len(PLOT_COLS)      # 4

    # Derive palette from the first available result
    first_df = next(iter(results_by_seat.values()))
    scales    = sorted(first_df["cost_scale"].unique())
    palette   = sns.color_palette("coolwarm_r", len(scales))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=120,
        squeeze=False,
    )
    fig.suptitle(city_pair.replace("_", " "), fontsize=label_fontsize + 2, y=1.01)

    legend_ax = None   # we'll put the legend on the last cell of the last row

    for r, num_seats in enumerate(seat_list):
        result    = results_by_seat[num_seats]
        fleet_min = int(result["fleet_size"].min())
        fleet_max = int(result["fleet_size"].max())

        for c, (col, col_label) in enumerate(PLOT_COLS):
            ax        = axes[r, c]
            is_legend = (r == n_rows - 1) and (c == n_cols - 1)

            sns.lineplot(
                data=result, x="fleet_size", y=col,
                hue="cost_scale", marker="o",
                palette=palette, ax=ax,
                legend=is_legend, markersize=5, linewidth=0.8,
            )

            # Column header on top row only
            if r == 0:
                ax.set_title(col_label, fontsize=label_fontsize)

            # x-axis label on bottom row only
            ax.set_xlabel("Fleet Size" if r == n_rows - 1 else "",
                          fontsize=label_fontsize)
            if r < n_rows - 1:
                ax.set_xticklabels([])

            # y-axis label on left column only
            ax.set_ylabel(f"{num_seats}-seat" if c == 0 else "",
                          fontsize=label_fontsize)

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

            # Profit axis: show thousands
            if col == "profit" or col == "revenue":
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{int(x / 1000)}")
                )

            ax.set_xlim(fleet_min, fleet_max)
            ax.xaxis.set_major_locator(
                MultipleLocator(max(1, (fleet_max - fleet_min) // 6))
            )
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
            ax.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)

            if is_legend:
                legend_ax = ax

    # Shared legend outside the grid
    if legend_ax is not None:
        handles, lbls = legend_ax.get_legend_handles_labels()
        legend_ax.get_legend().remove()
        scale_labels = [f"x{float(l):.2f}" for l in lbls]
        fig.legend(
            handles, scale_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            title="Cost Scale\n(x normal)",
            title_fontsize=tick_fontsize + 2,
            fontsize=tick_fontsize,
            borderaxespad=0.,
        )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved - {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plotting  (Plot_2 / Plot_3: RASM or CASM vs cost scale, cross-market)
# ---------------------------------------------------------------------------

def _market_label(city_pair: str) -> str:
    """'TAMU_Houston_Thu' -> 'TAMU-Houston Market'"""
    parts = city_pair.split("_")
    return "-".join(parts[:-1]) + " Market"


def _plot_unit_metric(
    results: dict,
    fleet_size: int,
    metric: str,
    y_label: str,
    title: str,
    output_path: Path | None = None,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """
    1-row x 3-col figure (cols = seat configs, shared y-axis).
    Lines = markets.  x-axis = cost scale (%).
    """
    city_pairs = list(results.keys())
    seat_list  = sorted(next(iter(results.values()))["results_by_seat"].keys())
    n_cols = len(seat_list)

    palette   = sns.color_palette("tab10", len(city_pairs))
    color_map = {cp: palette[i] for i, cp in enumerate(city_pairs)}

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(5 * n_cols, 5),
        dpi=120,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(f"{title}  |  Fleet = {fleet_size}",
                 fontsize=label_fontsize + 2, y=1.02)

    for c, num_seats in enumerate(seat_list):
        ax = axes[0, c]
        for city_pair in city_pairs:
            df = results[city_pair]["results_by_seat"].get(num_seats)
            if df is None:
                continue
            subset = df[df["fleet_size"] == fleet_size].sort_values("cost_pct")
            if subset.empty:
                continue
            ax.plot(
                subset["cost_pct"], subset[metric],
                marker="o", markersize=5, linewidth=1.2,
                color=color_map[city_pair],
                label=_market_label(city_pair),
            )

        ax.set_title(f"{num_seats}-seat", fontsize=label_fontsize)
        ax.set_xlabel("Cost Scale (%)", fontsize=label_fontsize)
        ax.set_ylabel(y_label if c == 0 else "", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.grid(True, linestyle="--", alpha=0.5)

    handles = [
        plt.Line2D([0], [0], color=color_map[cp], marker="o",
                   linewidth=1.2, markersize=5, label=_market_label(cp))
        for cp in city_pairs
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Market",
        title_fontsize=tick_fontsize + 2,
        fontsize=tick_fontsize,
        borderaxespad=0.,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved - {output_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_rasm(
    results: dict,
    fleet_size: int,
    output_path: Path | None = None,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """RASM vs cost scale across markets at a fixed fleet size."""
    _plot_unit_metric(
        results, fleet_size,
        metric="rasm", y_label="RASM ($/ASM)",
        title="RASM at Varying Cost Scale",
        output_path=output_path,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )


def plot_casm(
    results: dict,
    fleet_size: int,
    output_path: Path | None = None,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """CASM (revenue flights only) vs cost scale across markets at a fixed fleet size."""
    _plot_unit_metric(
        results, fleet_size,
        metric="casm_rev", y_label="CASM ($/ASM)",
        title="CASM at Varying Cost Scale",
        output_path=output_path,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )


# ---------------------------------------------------------------------------
# Plotting  (Plot_4: Sweet Spot Heatmap — profit by fleet x seat at fixed cost scales)
# ---------------------------------------------------------------------------

def plot_profit_heatmap(
    results_by_seat: dict,          # {num_seats: DataFrame}  from results[city_pair]["results_by_seat"]
    cost_scales: list,              # e.g. [0.4, 0.8, 1.2]
    city_pair: str = "",
    output_path: Path | None = None,
    label_fontsize: int = 14,
    tick_fontsize: int = 11,
):
    """
    Sweet Spot Heatmap: one subplot per cost_scale value.
    Each subplot is a heatmap with fleet size on x-axis and seat capacity on y-axis.
    Cell colour = operating profit ($).

    Usage:
        plot_profit_heatmap(
            results[city_pair]["results_by_seat"],
            cost_scales=[0.4, 0.8, 1.2],
            city_pair=city_pair,
        )
    """
    seat_list  = sorted(results_by_seat.keys())
    n_cols     = len(cost_scales)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(6 * n_cols, 4),
        dpi=120,
        squeeze=False,
    )
    title = f"Operating Profit Sweet Spot"
    if city_pair:
        title += f"  |  {_market_label(city_pair)}"
    fig.suptitle(title, fontsize=label_fontsize + 2, y=1.02)

    # Compute shared colour scale across all subplots
    all_profits = []
    for c, cs in enumerate(cost_scales):
        ax  = axes[0, c]
        pct = round(cs * 100)

        # Build matrix: rows=seats (ascending), cols=fleet sizes
        fleet_sizes = sorted(results_by_seat[seat_list[0]]["fleet_size"].unique())
        matrix = []
        for num_seats in seat_list:
            df     = results_by_seat[num_seats]
            subset = df[df["cost_pct"] == pct].set_index("fleet_size")["profit"]
            row    = [subset.get(f, np.nan) for f in fleet_sizes]
            matrix.append(row)

        heat_df = pd.DataFrame(
            matrix,
            index=[f"{s}-seat" for s in seat_list],
            columns=fleet_sizes,
        )

        annot_df = heat_df / 1000.0
        vmin = np.nanmin(heat_df.values)
        vmax = np.nanmax(heat_df.values)

        sns.heatmap(
            heat_df,
            ax=ax,
            vmin=vmin, vmax=vmax,
            cmap="RdYlGn",
            annot=annot_df,
            fmt=".1f",
            annot_kws={"size": tick_fontsize - 2},
            linewidths=0.4,
            cbar=True,
        )

        ax.set_title(f"Cost Scale {pct}%", fontsize=label_fontsize)
        ax.set_xlabel("Fleet Size", fontsize=label_fontsize)
        ax.set_ylabel("Seat Capacity" if c == 0 else "", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved - {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plotting  (Plot_5: Sweet Spot Heatmap — all markets, rows=city pairs, cols=cost scales)
# ---------------------------------------------------------------------------

def plot_profit_heatmap_all(
    results: dict,                  # {city_pair: {"results_by_seat": {num_seats: DataFrame}}}
    cost_scales: list,              # e.g. [0.4, 0.8, 1.2]
    output_path: Path | None = None,
    label_fontsize: int = 14,
    tick_fontsize: int = 11,
):
    """
    Sweet Spot Heatmap across all markets.
    Rows = city pairs, cols = cost scales.
    Each cell = heatmap of profit (fleet size x seat capacity).
    Each heatmap has its own colour scale.

    Usage:
        plot_profit_heatmap_all(
            results,
            cost_scales=[0.4, 0.8, 1.2],
        )
    """
    city_pairs = list(results.keys())
    n_rows     = len(city_pairs)
    n_cols     = len(cost_scales)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        dpi=120,
        squeeze=False,
    )

    for r, city_pair in enumerate(city_pairs):
        results_by_seat = results[city_pair]["results_by_seat"]
        seat_list       = sorted(results_by_seat.keys())
        fleet_sizes     = sorted(results_by_seat[seat_list[0]]["fleet_size"].unique())

        for c, cs in enumerate(cost_scales):
            ax  = axes[r, c]
            pct = round(cs * 100)

            matrix = []
            for num_seats in seat_list:
                df     = results_by_seat[num_seats]
                subset = df[df["cost_pct"] == pct].set_index("fleet_size")["profit"]
                row    = [subset.get(f, np.nan) for f in fleet_sizes]
                matrix.append(row)

            heat_df  = pd.DataFrame(
                matrix,
                index=[f"{s}-seat" for s in seat_list],
                columns=fleet_sizes,
            )
            annot_df = heat_df / 1000.0
            vmin     = np.nanmin(heat_df.values)
            vmax     = np.nanmax(heat_df.values)

            sns.heatmap(
                heat_df,
                ax=ax,
                vmin=vmin, vmax=vmax,
                cmap="RdYlGn",
                annot=annot_df,
                fmt=".1f",
                annot_kws={"size": tick_fontsize - 2},
                linewidths=0.4,
                cbar=True,
            )

            # Top row: cost scale header
            if r == 0:
                ax.set_title(f"Cost Scale {pct}%", fontsize=label_fontsize)
            else:
                ax.set_title("")

            # Bottom row: fleet size label
            ax.set_xlabel("Fleet Size" if r == n_rows - 1 else "",
                          fontsize=label_fontsize)
            if r < n_rows - 1:
                ax.set_xticklabels([])

            # Left col: market name as y-axis label
            ax.set_ylabel(_market_label(city_pair) if c == 0 else "",
                          fontsize=label_fontsize)

            ax.tick_params(axis="both", labelsize=tick_fontsize)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved - {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plotting  (Plot_6: Fare box-and-whisker — grouped by market, x-axis = seat config)
# ---------------------------------------------------------------------------

def plot_fare_boxplot(
    results_dir: Path,
    fleet_size: int,
    price_scale: int,               # cost_pct integer, e.g. 100
    city_pairs: list,               # ordered list of city-pair names
    seat_list: list | None = None,  # defaults to [4, 6, 8]
    output_path: Path | None = None,
    label_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """
    Box-and-whisker plot of fare across seat configurations and markets.

    X-axis   : seat configurations (default 4, 6, 8)
    Grouping : three boxes per x-position, one per market in city_pairs
    Y-axis   : fare ($)

    Raw f{fleet_size}_p{price_scale}.csv files are read directly; each row's
    `fare` value is one observation.  `passenger_arrival_time_slot` is retained
    for reference but all time slots are pooled together in the distribution.

    Args:
        results_dir : path to the case results folder
        fleet_size  : fleet size to select (e.g. 10)
        price_scale : cost-percentage tag in the filename (e.g. 100 → p100)
        city_pairs  : list of city-pair subfolder names in display order
        seat_list   : seat configurations to show on x-axis
    """
    if seat_list is None:
        seat_list = [4, 6, 8]

    n_seats   = len(seat_list)
    n_markets = len(city_pairs)

    palette    = sns.color_palette("tab10", n_markets)
    group_w    = 0.7                        # total width per x position
    box_w      = group_w / n_markets * 0.9  # individual box width
    offsets    = np.linspace(
        -group_w / 2 + box_w / 2,
        group_w / 2 - box_w / 2,
        n_markets,
    )

    fig, ax = plt.subplots(figsize=(3 * n_seats + 2, 5), dpi=120)

    legend_patches = []
    for m_idx, city_pair in enumerate(city_pairs):
        color = palette[m_idx]
        legend_patches.append(
            plt.matplotlib.patches.Patch(facecolor=color, label=_market_label(city_pair))
        )

        for s_idx, num_seats in enumerate(seat_list):
            csv_path = (
                results_dir / city_pair / f"{num_seats}seats"
                / f"f{fleet_size}_p{price_scale}.csv"
            )
            if not csv_path.exists():
                print(f"[plot_fare_boxplot] Missing {csv_path}, skipping")
                continue

            df   = pd.read_csv(csv_path, usecols=["fare", "passenger_arrival_time_slot"])
            data = df["fare"].dropna().values
            if len(data) == 0:
                continue

            x_pos = s_idx + 1 + offsets[m_idx]
            bp = ax.boxplot(
                data,
                positions=[x_pos],
                widths=box_w,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.6),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markersize=3, alpha=0.4, linestyle="none"),
                manage_ticks=False,
            )

    ax.set_xticks(range(1, n_seats + 1))
    ax.set_xticklabels([f"{s}-seat" for s in seat_list], fontsize=tick_fontsize)
    ax.set_xlabel("Seat Configuration", fontsize=label_fontsize)
    ax.set_ylabel("Fare ($)", fontsize=label_fontsize)
    ax.set_title(
        f"Fare Distribution  |  Fleet = {fleet_size}  |  Cost Scale = {price_scale}%",
        fontsize=label_fontsize,
    )
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(
        handles=legend_patches,
        title="Market",
        title_fontsize=tick_fontsize + 1,
        fontsize=tick_fontsize,
        loc="upper right",
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"[post_processing] Saved - {output_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_results(
    results_dir: Path,
    city_pair: str,
    vertiport_config: str,
    cfg: dict,
    root: Path,
    seat_filter: int | None = None,
) -> dict:
    """Load all seat results for one city pair. Returns {num_seats: DataFrame}."""
    with open(root / vertiport_config) as f:
        vp_cfg = json.load(f)
    ft_matrix = np.array(vp_cfg["links"]["flight_time_matrix"],     dtype=float)
    fd_matrix = np.array(vp_cfg["links"]["flight_distance_matrix"], dtype=float)

    cost_cfg = cfg["cost"]
    base_out = root / results_dir / city_pair
    seat_list = cfg["num_seats"]
    if seat_filter is not None:
        seat_list = [s for s in seat_list if s == seat_filter]

    results_by_seat = {}
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
        results_by_seat[num_seats] = result
    return results_by_seat


def main():
    parser = argparse.ArgumentParser(description="Post-process and visualize RAM pricing results")
    parser.add_argument("--results_dir",      required=True,
                        help="Base results directory (e.g. data/results/case3_costScalingFrom4Seater_updated)")
    parser.add_argument("--city_pair",        required=True, action="append", dest="city_pairs",
                        help="City-pair subdirectory (repeatable for multi-market plots)")
    parser.add_argument("--vertiport_config", required=True, action="append", dest="vertiport_configs",
                        help="Vertiport config JSON matching each --city_pair (same order)")
    parser.add_argument("--optimizer_config", default="configs/optimizer.json",
                        help="Shared optimizer config JSON (default: configs/optimizer.json)")
    parser.add_argument("--seats", type=int, default=None,
                        help="Only process this seat count (default: all)")
    parser.add_argument("--fleet_size", type=int, default=None,
                        help="If set, produce a RASM/CASM vs cost-scale plot at this fleet size")
    parser.add_argument("--label_fontsize", type=int, default=16,
                        help="Font size for axis labels and column headers (default: 16)")
    parser.add_argument("--tick_fontsize",  type=int, default=12,
                        help="Font size for tick labels and legend (default: 12)")
    args = parser.parse_args()

    if len(args.city_pairs) != len(args.vertiport_configs):
        raise ValueError("Each --city_pair must have a matching --vertiport_config")

    root = Path(__file__).parent

    with open(root / args.optimizer_config) as f:
        cfg = json.load(f)

    plots_dir = root / args.results_dir / "summary_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for city_pair, vp_config in zip(args.city_pairs, args.vertiport_configs):
        results_by_seat = load_results(
            Path(args.results_dir), city_pair, vp_config, cfg, root, args.seats
        )
        if results_by_seat:
            results[city_pair] = {"results_by_seat": results_by_seat}

    if not results:
        print("[post_processing] No results loaded.")
        return

    if args.fleet_size is not None:
        plot_rasm(
            results, args.fleet_size,
            output_path=plots_dir / f"rasm_fleet{args.fleet_size}.png",
            label_fontsize=args.label_fontsize,
            tick_fontsize=args.tick_fontsize,
        )
        plot_casm(
            results, args.fleet_size,
            output_path=plots_dir / f"casm_fleet{args.fleet_size}.png",
            label_fontsize=args.label_fontsize,
            tick_fontsize=args.tick_fontsize,
        )
    else:
        for city_pair, data in results.items():
            output_path = plots_dir / f"{city_pair}_summary.png"
            plot_market(
                data["results_by_seat"], city_pair, output_path,
                label_fontsize=args.label_fontsize,
                tick_fontsize=args.tick_fontsize,
            )

    print("[post_processing] Done.")


if __name__ == "__main__":
    main()
