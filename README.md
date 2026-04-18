# wisk_ram_study

## Running the pipeline

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
# Weekday
python run_pipeline.py --state Illinois --o_city Chicago --d_city UIUC \
    --vertiport_config config/vertiport_configuration/UIUC.json --day Thu

# Weekend
python run_pipeline.py --state Florida --o_city UFL --d_city Orlando \
    --vertiport_config config/vertiport_configuration/Orlando.json --day Sat
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

## Running the pricing optimization

Requires demand files from the pipeline above. Reads the vertiport config directly — no hardcoded matrices.

```bash
python run_pricing.py \
    --vertiport_config external/replica_data_analytics/config/vertiport_configuration/UFL.json \
    --demand_dir data/processed_data/demand \
    --output_dir data/results/UFL_Orlando
```

**Required args:**

| Arg | Description |
|-----|-------------|
| `--vertiport_config` | Path to vertiport JSON config (must have a `links` section) |
| `--demand_dir` | Folder containing `*_trips.csv` files from `run_pipeline.py` |
| `--output_dir` | Where to write optimization results |

**Optional sweep/optimizer args** (defaults shown):

| Arg | Default | Description |
|-----|---------|-------------|
| `--fleet_min/max/step` | 10 / 65 / 5 | Fleet size sweep range |
| `--casm_min/max/step` | 0.6 / 1.6 / 0.1 | Cost per available seat mile sweep |
| `--optimality_gap` | 0.05 | Gurobi MIP gap |
| `--time_limit` | 3600 | Gurobi time limit (seconds) |
| `--utility_type` | `betas` | Utility model: `betas` or `vot` |

Before running, the vertiport config must have a `links` section defining each OD pair with `distance_miles` and `flight_time_min`. A verification check runs automatically at startup — it will report mismatches between demand files and config before touching the optimizer.

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
