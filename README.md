# wisk_ram_study

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd wisk_ram_study
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Create the virtual environment

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 recommended.

```bash
uv venv wisk --python 3.11
```

Activate:

```bash
# Windows
wisk\Scripts\activate

# macOS / Linux
source wisk/bin/activate
```

### 3. Install dependencies

```bash
uv sync --group dev
```

> **Note:** activating the venv does not install packages. You must run `uv sync` explicitly. Re-run it whenever `pyproject.toml` changes (e.g. after a teammate adds a new dependency).

---

## Running the input data pipeline

Data relies on raw Replica data. For RAM analysis, the demand data is aggregated from disaggregated data in Replica.

```bash
python run_pipeline.py \
    --state <state> \
    --o_city <origin_city> \
    --d_city <destination_city> \
    --vertiport_config <path_to_config.json> \
    --day <Thu|Sat>
```

**Arguments:**

| Arg | Required | Description |
|-----|----------|-------------|
| `--state` | Yes | State name (e.g. `Illinois`, `Florida`) |
| `--o_city` | Yes | Origin city (e.g. `Chicago`, `UFL`) |
| `--d_city` | Yes | Destination city (e.g. `UIUC`, `Orlando`) |
| `--vertiport_config` | Yes | Path to vertiport JSON config |
| `--day` | No | Day of week: `Thu` or `Sat` (default: `Thu`) |

**Examples:**
```bash
# Illinois — Chicago ↔ UIUC (weekday)
python run_pipeline.py --state Illinois --o_city Chicago --d_city UIUC \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UIUC.json \
    --day Thu

# Florida — UFL ↔ Orlando (weekday)
python run_pipeline.py --state Florida --o_city UFL --d_city Orlando \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json \
    --day Thu
```

The pipeline will:
1. Check if OD cost matrices already exist for each city in `data/processed_data/od_cost_matrix/`. If not, generate and cache them from the raw Replica data.
2. Run the multimodal trip generation using the cached matrices.
3. Save trip demand output to `data/processed_data/demand/`.

**Data layout:**
```
data/
├── replica_data/<state>/<o_city>_<d_city>_<Thu|Sat>.csv   ← raw input
└── processed_data/
    ├── od_cost_matrix/<state>/
    │   ├── <city>_Thu_hourly_od_lookup_tables/             ← cached weekday OD matrices
    │   ├── <city>_Sat_hourly_od_lookup_tables/             ← cached weekend OD matrices
    │   └── od_distance_matrix/
    └── demand/
        ├── UFL_Orlando_Thu/                                ← trip output per route+day
        ├── UFL_Orlando_Sat/
        └── ...
```


## Running the pricing optimization

Requires demand files produced by the pipeline above.

### Quick start

```bash
python run_pricing.py \
    --city_pair        UFL_Orlando_Thu \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json
```

```bash
# Illinois example
python run_pricing.py \
    --city_pair        Chicago_UIUC_Thu \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UIUC.json
```

Both travel directions (`UFL→Orlando` **and** `Orlando→UFL`) are loaded automatically
from `data/processed_data/demand/` — no need to specify a demand path.

**CLI arguments:**

| Arg | Required | Description |
|-----|----------|-------------|
| `--city_pair` | Yes | `{o_city}_{d_city}_{day}` — both directions loaded automatically (e.g. `UFL_Orlando_Thu`) |
| `--vertiport_config` | Yes | Path to vertiport JSON config |
| `--optimizer_config` | No | Shared optimizer config (default: `configs/optimizer.json`) |

### Shared optimizer config — `configs/optimizer.json`

All sweep and optimizer parameters live in one shared file used by every study.
Edit this file to change the sweep without touching the runner script.

```jsonc
{
  "output_base_dir": "data/results",       // root output directory

  "optimizer": {
    "start_hour":      6,                  // ignore demand before this hour
    "time_resolution": 30,                 // scheduling slot length (minutes)
    "value_of_time":   32.63,              // passenger value of time ($/hr)
    "beta_cost":       0.0353,             // cost coefficient for utility model
    "utility_type":    "betas",            // "betas" or "vot"
    "optimality_gap":  0.05,               // MIP optimality gap (5 %)
    "time_limit":      3600                // solver wall-clock limit (seconds)
  },

  "fleet_sweep": { "min": 10, "max": 65, "step": 5 },

  "cost_steps": 10,                        // number of cost points between min and max

  "seat_configs": [
    {
      "num_seats":   4,
      "cost_per_fh": { "min": 200, "max": 500 },   // $/flight-hour range
      "cost_per_fc": { "min":  20, "max":  50 }    // $/flight-cycle range
    },
    { "num_seats": 6, "cost_per_fh": { "min": 260, "max": 650 }, "cost_per_fc": { "min": 26, "max": 65 } },
    { "num_seats": 8, "cost_per_fh": { "min": 320, "max": 800 }, "cost_per_fc": { "min": 32, "max": 80 } }
  ]
}
```

`uam_transition_time` (boarding/deboarding OVTT) and per-vertiport `fato_capacity` are
read directly from the vertiport config — see [Vertiport config format](#vertiport-config-format) below.

### Output layout

Results are written under `{output_base_dir}/{city_pair}/{num_seats}seats/`:

```
data/results/
└── UFL_Orlando_Thu/
    ├── 4seats/
    │   ├── f10_p0.csv          ← fleet 10, min-cost point  (0 %)
    │   ├── f10_p25.csv         ← fleet 10, 25 % of cost range
    │   ├── f10_p50.csv
    │   ├── f10_p75.csv
    │   ├── f10_p100.csv        ← fleet 10, max-cost point (100 %)
    │   ├── repo_f10_p0.csv     ← repositioning flights for the same run
    │   └── ...
    ├── 6seats/
    └── 8seats/
```

File naming: `f{fleet}_p{cost_pct}.csv` where `cost_pct` is the integer percentage
(0–100) along the `[min, max]` cost range defined in `optimizer.json`.

### Vertiport config format

Each vertiport config (under `external/replica_data_analytics/config/vertiport_configuration/`)
must contain:

- `assumptions.multimodal_segments.option1_ram` — the `middle_mile` segment's `ovtt_min`
  is used as `uam_transition_time` (boarding + deboarding overhead in minutes).
- `vertiports[*].fato_capacity` — per-vertiport FATO pad capacity (integer), passed to
  the optimizer as a list ordered by `links.nodes`.
- `links.nodes` — ordered list of vertiport IDs (defines row/column order for all matrices).
- `links.flight_time_matrix` — N×N flight time matrix (minutes).
- `links.flight_distance_matrix` — N×N flight distance matrix (miles).

### Visualising results

```bash
python visualize_results.py \
    --city_pair        UFL_Orlando_Thu \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json

# Limit to one seat configuration
python visualize_results.py \
    --city_pair        UFL_Orlando_Thu \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json \
    --seats 4
```

Saves one `summary_plot.png` per seat-capacity subdirectory.

---

## Submodules

| Path | Repo | Branch |
|------|------|--------|
| `external/replica_data_analytics` | [AndrewParkJH/replica_data_analytics](https://github.com/AndrewParkJH/replica_data_analytics) | default |
| `external/uam_system_model` | [caoalbert/uam_system_model](https://github.com/caoalbert/uam_system_model) | `vertiport_capacity` |

To update submodules to their latest tracked commits:

```bash
git submodule update --remote
```
