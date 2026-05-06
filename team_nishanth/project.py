import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import pandas as pd
from team_nishanth.helper import (
    load_csv,
    save_csv,
    save_json,
    ensure_directory_exists,
    compute_missing_percentage,
)

METADATA_COLUMNS = ["updated_at"]
IDENTIFIER_COLUMNS = ["year", "country"]


def load_raw_indicators(raw_data_path):
    """
    Loads the raw World Bank indicators CSV into a DataFrame.

    Args:
        raw_data_path (str): Path to the raw world_bank_indicators.csv file.

    Returns:
        pd.DataFrame: Raw indicators DataFrame with all original columns.
    """
    df = load_csv(raw_data_path)
    print(f"[load] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def drop_metadata_columns(df):
    """
    Drops administrative/metadata columns that carry no analytical value.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with metadata columns removed.
    """
    cols_to_drop = [c for c in METADATA_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[drop_metadata] Dropped columns: {cols_to_drop}")
    return df


def standardize_country_codes(df):
    """
    Converts the 'country' column to uppercase ISO-3 codes for consistency.

    Args:
        df (pd.DataFrame): DataFrame containing a 'country' column.

    Returns:
        pd.DataFrame: DataFrame with uppercased country codes.
    """
    df = df.copy()
    df["country"] = df["country"].str.upper()
    print("[standardize] Country codes converted to uppercase.")
    return df


def analyze_and_report_missing(df):
    """
    Computes and prints a missing-value report for all indicator columns.

    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        dict: Mapping of column name to missing percentage (0.0–1.0),
              sorted descending by missingness.
    """
    missing_pct = compute_missing_percentage(df)
    indicator_missing = {
        col: pct
        for col, pct in missing_pct.items()
        if col not in IDENTIFIER_COLUMNS
    }
    sorted_missing = dict(
        sorted(indicator_missing.items(), key=lambda x: x[1], reverse=True)
    )
    high_missing = sum(1 for v in sorted_missing.values() if v > 0.5)
    print(
        f"[missing] {high_missing} / {len(sorted_missing)} indicator columns "
        f"have >50% missing values."
    )
    return sorted_missing


def drop_sparse_columns(df, missing_report, threshold):
    """
    Removes indicator columns whose missing-value percentage exceeds the threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        missing_report (dict): Output of analyze_and_report_missing.
        threshold (float): Drop columns with missing percentage > this value (0.0–1.0).

    Returns:
        tuple[pd.DataFrame, list[str]]:
            - DataFrame with sparse columns removed.
            - List of dropped column names.
    """
    cols_to_drop = [
        col for col, pct in missing_report.items() if pct > threshold
    ]
    df = df.drop(columns=cols_to_drop)
    print(
        f"[drop_sparse] Dropped {len(cols_to_drop)} columns above "
        f"{threshold:.0%} missing threshold. {len(df.columns)} columns remain."
    )
    return df, cols_to_drop


def impute_missing_values(df):
    """
    Fills remaining missing values in indicator columns using a two-pass strategy:
      1. Within each country group, forward-fill then backward-fill by year
         (preserves time-series continuity).
      2. Any still-missing values are filled with the column-wide median
         (handles countries with no data at all for that indicator).

    Args:
        df (pd.DataFrame): DataFrame after sparse columns have been dropped.

    Returns:
        pd.DataFrame: DataFrame with no missing values in indicator columns.
    """
    df = df.copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    indicator_cols = [c for c in df.columns if c not in IDENTIFIER_COLUMNS]

    df[indicator_cols] = (
        df.groupby("country")[indicator_cols]
        .transform(lambda group: group.ffill().bfill())
    )

    col_medians = df[indicator_cols].median()
    df[indicator_cols] = df[indicator_cols].fillna(col_medians)

    remaining_nulls = df[indicator_cols].isna().sum().sum()
    print(f"[impute] Imputation complete. Remaining nulls: {remaining_nulls}")
    return df


def normalize_numeric_columns(df):
    """
    Applies min-max normalization (scale to [0, 1]) to all numeric indicator columns.
    The 'year' column is excluded from normalization.

    Args:
        df (pd.DataFrame): DataFrame after imputation.

    Returns:
        tuple[pd.DataFrame, dict]:
            - Normalized DataFrame.
            - Scaler parameters dict: {column: {"min": float, "max": float}}
              for reproducibility and inverse transformation.
    """
    df = df.copy()
    indicator_cols = [
        c for c in df.columns
        if c not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    scaler_params = {}
    for col in indicator_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        scaler_params[col] = {"min": col_min, "max": col_max}
        col_range = col_max - col_min
        if col_range == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / col_range

    print(f"[normalize] Min-max normalization applied to {len(indicator_cols)} columns.")
    return df, scaler_params


def run_preprocessing_pipeline(config):
    """
    Orchestrates the full preprocessing pipeline for World Bank indicator data.

    Steps:
      1. Load raw CSV.
      2. Drop metadata columns (updated_at).
      3. Standardize country codes to uppercase.
      4. Analyze missing values.
      5. Drop columns exceeding the missing-value threshold.
      6. Impute remaining missing values (forward/backward fill + median).
      7. Normalize numeric indicator columns to [0, 1].
      8. Save cleaned CSV, scaler parameters, and dropped-columns list.

    Args:
        config (dict): Must contain keys:
            - raw_data_path (str): Path to raw world_bank_indicators.csv.
            - cleaned_data_dir (str): Directory to write outputs.
            - output_file_name (str): Filename for the cleaned CSV.
            - scaler_params_file (str): Filename for scaler params JSON.
            - dropped_columns_file (str): Filename for dropped columns JSON.
            - missing_value_threshold (float): Column drop threshold (0.0–1.0).

    Returns:
        None
    """
    ensure_directory_exists(config["cleaned_data_dir"])

    df = load_raw_indicators(config["raw_data_path"])
    df = drop_metadata_columns(df)
    df = standardize_country_codes(df)

    missing_report = analyze_and_report_missing(df)
    df, dropped_cols = drop_sparse_columns(
        df, missing_report, config["missing_value_threshold"]
    )

    df = impute_missing_values(df)
    df, scaler_params = normalize_numeric_columns(df)

    cleaned_csv_path = os.path.join(
        config["cleaned_data_dir"], config["output_file_name"]
    )
    scaler_path = os.path.join(
        config["cleaned_data_dir"], config["scaler_params_file"]
    )
    dropped_path = os.path.join(
        config["cleaned_data_dir"], config["dropped_columns_file"]
    )

    save_csv(df, cleaned_csv_path)
    save_json(scaler_params, scaler_path)
    save_json(dropped_cols, dropped_path)

    print(f"\n[done] Cleaned data saved to:    {cleaned_csv_path}")
    print(f"[done] Scaler params saved to:   {scaler_path}")
    print(f"[done] Dropped columns saved to: {dropped_path}")
    print(f"[done] Final shape: {df.shape[0]} rows × {df.shape[1]} columns.")


def add_lag_features(df, lag_periods):
    """
    Adds lag features for each numeric indicator column, grouped by country.
    For each lag n, a new column '{col}_lag_{n}' contains that column's value n years prior.
    Rows at the start of each country's time series will be NaN (no prior data exists).

    Args:
        df (pd.DataFrame): Cleaned, normalized indicator DataFrame.
        lag_periods (list[int]): List of lag offsets in years, e.g. [1, 2, 5].

    Returns:
        pd.DataFrame: DataFrame with lag feature columns appended.
    """
    df = df.copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    indicator_cols = [
        c for c in df.columns
        if c not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in indicator_cols:
        for lag in lag_periods:
            df[f"{col}_lag_{lag}"] = df.groupby("country")[col].shift(lag)
    total_added = len(indicator_cols) * len(lag_periods)
    print(
        f"[lag_features] Added {total_added} lag columns "
        f"(lags {lag_periods} on {len(indicator_cols)} indicators)."
    )
    return df


def add_rolling_averages(df, rolling_windows):
    """
    Adds rolling mean features for each numeric indicator column, grouped by country.
    For each window w, a new column '{col}_rolling_mean_{w}' contains the mean of the
    past w years. Uses min_periods=1 so early rows use however many years are available.
    Only applied to original indicator columns, not to any existing lag columns.

    Args:
        df (pd.DataFrame): DataFrame (may already contain lag columns).
        rolling_windows (list[int]): List of window sizes in years, e.g. [3, 5].

    Returns:
        pd.DataFrame: DataFrame with rolling mean columns appended.
    """
    df = df.copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    indicator_cols = [
        c for c in df.columns
        if c not in IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and "_lag_" not in c
        and "_rolling_mean_" not in c
    ]
    for col in indicator_cols:
        for window in rolling_windows:
            df[f"{col}_rolling_mean_{window}"] = (
                df.groupby("country")[col]
                .transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
            )
    total_added = len(indicator_cols) * len(rolling_windows)
    print(
        f"[rolling_avg] Added {total_added} rolling mean columns "
        f"(windows {rolling_windows} on {len(indicator_cols)} indicators)."
    )
    return df


def run_feature_engineering_pipeline(config):
    """
    Orchestrates the feature engineering pipeline.
    Loads the cleaned CSV, adds lag features and rolling averages, then saves
    the result as a new CSV in the same cleaned_data directory.

    Steps:
      1. Load cleaned CSV produced by run_preprocessing_pipeline.
      2. Add lag features for each configured lag period.
      3. Add rolling mean features for each configured window size.
      4. Save the feature-engineered DataFrame to features_output_file_name.

    Args:
        config (dict): Must contain keys:
            - cleaned_data_dir (str): Directory containing the cleaned CSV.
            - output_file_name (str): Filename of the cleaned CSV to load.
            - features_output_file_name (str): Filename for the output features CSV.
            - lag_periods (list[int]): Lag offsets in years.
            - rolling_windows (list[int]): Rolling window sizes in years.

    Returns:
        None
    """
    cleaned_csv_path = os.path.join(config["cleaned_data_dir"], config["output_file_name"])
    df = load_csv(cleaned_csv_path)
    print(f"[feature_eng] Loaded cleaned data: {df.shape[0]} rows × {df.shape[1]} columns.")

    df = add_lag_features(df, config["lag_periods"])
    df = add_rolling_averages(df, config["rolling_windows"])

    features_path = os.path.join(
        config["cleaned_data_dir"], config["features_output_file_name"]
    )
    save_csv(df, features_path)

    print(f"\n[done] Feature-engineered data saved to: {features_path}")
    print(f"[done] Final shape: {df.shape[0]} rows × {df.shape[1]} columns.")
