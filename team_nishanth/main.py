import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

from team_nishanth.project import run_preprocessing_pipeline


def load_config():
    """
    Loads environment variables from .env and returns a config dictionary.

    Returns:
        dict: Configuration values with keys:
            - raw_data_path (str)
            - cleaned_data_dir (str)
            - output_file_name (str)
            - scaler_params_file (str)
            - dropped_columns_file (str)
            - missing_value_threshold (float)
    """
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    return {
        "raw_data_path": os.getenv("RAW_DATA_PATH"),
        "cleaned_data_dir": os.getenv("CLEANED_DATA_DIR"),
        "output_file_name": os.getenv("OUTPUT_FILE_NAME"),
        "scaler_params_file": os.getenv("SCALER_PARAMS_FILE"),
        "dropped_columns_file": os.getenv("DROPPED_COLUMNS_FILE"),
        "missing_value_threshold": float(os.getenv("MISSING_VALUE_THRESHOLD", 0.5)),
    }


def validate_config(config):
    """
    Validates that all required config values are present and the raw data file exists.

    Args:
        config (dict): Config dict returned by load_config.

    Raises:
        ValueError: If any required config key is missing or empty.
        FileNotFoundError: If the raw data file does not exist.
    """
    required_keys = [
        "raw_data_path",
        "cleaned_data_dir",
        "output_file_name",
        "scaler_params_file",
        "dropped_columns_file",
    ]
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required config value: {key}")

    if not os.path.exists(config["raw_data_path"]):
        raise FileNotFoundError(
            f"Raw data file not found: {config['raw_data_path']}"
        )

    threshold = config["missing_value_threshold"]
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"MISSING_VALUE_THRESHOLD must be between 0 and 1, got: {threshold}"
        )


def main():
    config = load_config()
    validate_config(config)

    print("Config loaded and validated.")
    print(f"  Raw data:  {config['raw_data_path']}")
    print(f"  Output dir: {config['cleaned_data_dir']}")
    print(f"  Missing threshold: {config['missing_value_threshold']:.0%}\n")

    run_preprocessing_pipeline(config)


if __name__ == "__main__":
    main()
