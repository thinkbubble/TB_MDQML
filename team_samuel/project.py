import pandas as pd
import numpy as np


def load_npz_into_dataframe(npz_file_path):
    data = np.load(npz_file_path, allow_pickle=True)

    df = pd.DataFrame(data["data"], columns=data["columns"])

    encodings = data["encodings"].item()

    df["filename"] = df["filename"].astype(int).map(
        lambda x: encodings["filename"]["classes"][x]
    )

    df["label"] = df["label"].astype(int).map(
        lambda x: encodings["label"]["classes"][x]
    )

    return df


def run_initial_classification_pipeline(raw_data_folder_path, cleaned_data_file_path):
    df = load_npz_into_dataframe(cleaned_data_file_path)

    print(df.head())
    print("\nShape:", df.shape)

    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y



run_initial_classification_pipeline("team_samuel/Data", "team_samuel/Data/cleaned_data/csv_features_30_sec_cleaned.npz")