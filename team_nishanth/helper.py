import os
import json
import pandas as pd


def load_csv(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def save_csv(df, file_path):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Destination path for the CSV file.

    Returns:
        None
    """
    df.to_csv(file_path, index=False)


def save_json(data, file_path):
    """
    Saves a dictionary or list to a JSON file.

    Args:
        data (dict | list): Data to serialize.
        file_path (str): Destination path for the JSON file.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def ensure_directory_exists(dir_path):
    """
    Creates a directory (and any intermediate directories) if it does not exist.

    Args:
        dir_path (str): Path to the directory to create.

    Returns:
        None
    """
    os.makedirs(dir_path, exist_ok=True)


def compute_missing_percentage(df):
    """
    Computes the percentage of missing values for each column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Mapping of column name to its missing value percentage (0.0 to 1.0).
    """
    total_rows = len(df)
    if total_rows == 0:
        return {col: 0.0 for col in df.columns}
    return {col: df[col].isna().sum() / total_rows for col in df.columns}
