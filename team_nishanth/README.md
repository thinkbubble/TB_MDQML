# team_nishanth — World Bank Indicator Preprocessing & Feature Engineering

## Project Purpose

Cleans, preprocesses, and engineers features from the World Bank Indicator dataset for downstream ML modeling.
The raw dataset (`world_bank_indicators.csv`) contains ~16,960 rows, 215 columns, covering
economic, health, climate, education, and other indicators across countries from 1960 onward.

The pipeline produces a cleaned CSV with no missing values, standardized country codes,
and min-max normalized numeric indicators — then extends it with lag and rolling mean features
ready for model training.

---

## How to Run

Run from the **repo root** (`TB_MDQML/`):

```bash
python -m team_nishanth.main
```

Outputs are written to `team_nishanth/cleaned_data/`.

---

## Environment Variables (`.env`)

All paths and thresholds are configured in `team_nishanth/.env`.
Update these values to run on a different machine — no code changes needed.

| Variable                    | Description                                              | Example value                                                                                          |
|-----------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `RAW_DATA_PATH`             | Path to raw world_bank_indicators.csv                    | `./team_nishanth/raw_data/world_bank_indicator_dataset/world_bank_indicator/world_bank_indicators.csv` |
| `CLEANED_DATA_DIR`          | Directory to write all output files                      | `./team_nishanth/cleaned_data`                                                                         |
| `OUTPUT_FILE_NAME`          | Filename for the cleaned CSV                             | `world_bank_indicators_cleaned.csv`                                                                    |
| `SCALER_PARAMS_FILE`        | Filename for the min-max scaler parameters JSON          | `scaler_params.json`                                                                                   |
| `DROPPED_COLUMNS_FILE`      | Filename for the list of columns dropped due to sparsity | `dropped_columns.json`                                                                                 |
| `MISSING_VALUE_THRESHOLD`   | Drop any column with more than this fraction missing     | `0.5`                                                                                                  |
| `FEATURES_OUTPUT_FILE_NAME` | Filename for the feature-engineered output CSV           | `world_bank_indicators_features.csv`                                                                   |
| `LAG_PERIODS`               | Comma-separated list of lag offsets in years             | `1,2,5`                                                                                                |
| `ROLLING_WINDOWS`           | Comma-separated list of rolling mean window sizes        | `3,5`                                                                                                  |

---

## End-to-End Pipeline Flow

```text
main.py
  └── load_config()                          Load .env into config dict
  └── validate_config()                      Check all keys present and valid
  └── run_preprocessing_pipeline(config)     [project.py]
        └── load_raw_indicators()            Read CSV from RAW_DATA_PATH
        └── drop_metadata_columns()          Remove 'updated_at'
        └── standardize_country_codes()      Uppercase ISO-3 codes
        └── analyze_and_report_missing()     Compute missingness per column
        └── drop_sparse_columns()            Drop cols > MISSING_VALUE_THRESHOLD missing
        └── impute_missing_values()          Forward/backward fill per country, then median
        └── normalize_numeric_columns()      Min-max scale all numeric indicators to [0,1]
        └── save_csv()                       Write cleaned CSV
        └── save_json()                      Write scaler_params.json
        └── save_json()                      Write dropped_columns.json
  └── run_feature_engineering_pipeline(config)   [project.py]
        └── load_csv()                       Load cleaned CSV
        └── add_lag_features()               Add {col}_lag_{n} columns per country
        └── add_rolling_averages()           Add {col}_rolling_mean_{w} columns per country
        └── save_csv()                       Write feature-engineered CSV
```

---

## Function Reference

### `main.py`

#### `load_config()`

- **Input:** None (reads from `team_nishanth/.env`)
- **Output:** `dict` — keys: `raw_data_path`, `cleaned_data_dir`, `output_file_name`, `scaler_params_file`, `dropped_columns_file`, `missing_value_threshold`, `features_output_file_name`, `lag_periods`, `rolling_windows`

#### `validate_config(config)`

- **Input:** `config` (dict)
- **Output:** None — raises `ValueError` if a required key is missing, threshold is out of range, lag periods are invalid, or rolling windows are invalid; raises `FileNotFoundError` if raw data file does not exist

---

### `project.py`

#### `load_raw_indicators(raw_data_path)`

- **Input:** `raw_data_path` (str) — path to the raw CSV
- **Output:** `pd.DataFrame` — full raw dataset (16,960 rows × 215 columns)

#### `drop_metadata_columns(df)`

- **Input:** `df` (pd.DataFrame)
- **Output:** `pd.DataFrame` — same data with `updated_at` removed

#### `standardize_country_codes(df)`

- **Input:** `df` (pd.DataFrame) — must contain a `country` column
- **Output:** `pd.DataFrame` — `country` values uppercased (e.g., `abw` → `ABW`)

#### `analyze_and_report_missing(df)`

- **Input:** `df` (pd.DataFrame)
- **Output:** `dict` — `{column_name: missing_fraction}` sorted descending; prints summary

#### `drop_sparse_columns(df, missing_report, threshold)`

- **Input:** `df` (pd.DataFrame), `missing_report` (dict), `threshold` (float, 0–1)
- **Output:** `tuple[pd.DataFrame, list[str]]` — cleaned DataFrame and list of dropped column names

#### `impute_missing_values(df)`

- **Input:** `df` (pd.DataFrame) — after sparse columns dropped
- **Output:** `pd.DataFrame` — no missing values; data sorted by `(country, year)`
- **Strategy:** forward-fill then backward-fill within each country group (time-series continuity), then column-wide median for any country with zero data for that indicator

#### `normalize_numeric_columns(df)`

- **Input:** `df` (pd.DataFrame) — after imputation
- **Output:** `tuple[pd.DataFrame, dict]` — normalized DataFrame and scaler params `{col: {min, max}}`
- **Strategy:** min-max scaling to [0, 1]; `year` and `country` are excluded

#### `run_preprocessing_pipeline(config)`

- **Input:** `config` (dict) — full config as returned by `load_config`
- **Output:** None — writes three files to `cleaned_data_dir`: cleaned CSV, scaler params JSON, dropped columns JSON

#### `add_lag_features(df, lag_periods)`

- **Input:** `df` (pd.DataFrame) — cleaned normalized DataFrame; `lag_periods` (list[int]) — lag offsets in years
- **Output:** `pd.DataFrame` — original columns plus `{col}_lag_{n}` for each indicator and each lag
- **Note:** Rows at the start of each country's time series will be NaN where no prior years exist

#### `add_rolling_averages(df, rolling_windows)`

- **Input:** `df` (pd.DataFrame) — may already contain lag columns; `rolling_windows` (list[int]) — window sizes in years
- **Output:** `pd.DataFrame` — original columns plus `{col}_rolling_mean_{w}` for each indicator and each window
- **Note:** Uses `min_periods=1` so early rows use however many years are available; applied to original indicator columns only, not to lag columns

#### `run_feature_engineering_pipeline(config)`

- **Input:** `config` (dict) — full config as returned by `load_config`
- **Output:** None — writes `features_output_file_name` to `cleaned_data_dir`

---

### `helper.py`

#### `load_csv(file_path)`

- **Input:** `file_path` (str)
- **Output:** `pd.DataFrame`

#### `save_csv(df, file_path)`

- **Input:** `df` (pd.DataFrame), `file_path` (str)
- **Output:** None

#### `save_json(data, file_path)`

- **Input:** `data` (dict | list), `file_path` (str)
- **Output:** None

#### `ensure_directory_exists(dir_path)`

- **Input:** `dir_path` (str)
- **Output:** None — creates directory (and parents) if not present

#### `compute_missing_percentage(df)`

- **Input:** `df` (pd.DataFrame)
- **Output:** `dict` — `{column_name: missing_fraction}` for every column
