import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

from preprocess_data import sensing_and_signal_ingestion
from team_amrutha.project import run_initial_classification_pipeline


def load_config():
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    return {
        "raw_data_folder_path": os.getenv("RAW_DATA_FOLDER_PATH"),
        "cleaned_data_folder_path": os.getenv("CLEANED_DATA_FOLDER_PATH"),
        "checkpoint_file_path": os.getenv("CHECKPOINT_FILE_PATH"),
    }


def validate_config(config):
    for key, value in config.items():
        if not value:
            raise ValueError(f"Missing config value: {key}")


def main():
    config = load_config()
    validate_config(config)

    master_unique_id, external_data_mapping, internal_data = sensing_and_signal_ingestion(
        config["raw_data_folder_path"],
        config["cleaned_data_folder_path"],
        config["checkpoint_file_path"]
    )

    print("MASTER_UNIQUE_ID:", master_unique_id)
    print("external_data_mapping:", external_data_mapping)
    print("internal_data:", internal_data)

    print("\nStarting initial classification model...\n")

    run_initial_classification_pipeline(
        raw_data_folder_path=config["raw_data_folder_path"],
        cleaned_data_folder_path=config["cleaned_data_folder_path"]
    )


if __name__ == "__main__":
    main()