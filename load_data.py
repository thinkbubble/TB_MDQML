

from helper import undo_one_hot_encoding
from helper import undo_label_encoding
from helper import read_from_json
import pandas as pd
import numpy as np
import string
import random
import os


# ================= TXT =================
def load_preprocessed_txt(npz_path):
    # Load preprocessed text tokens
    arr = np.load(npz_path, allow_pickle=True)
    tokens = arr["tokens"]
    return tokens.tolist()  # return as Python list of strings


# ================= CSV =================
def load_preprocessed_csv(npz_path, decode=False):
    # Load preprocessed CSV data from .npz
    # If decode=True, attempt to reconstruct original categorical labels
    #
    # Args:
    #     npz_path (str): Path to the .npz file created by preprocess_csv_files_in_folder
    #     decode (bool): Whether to decode categorical columns back to original labels
    #
    # Returns:
    #     pd.DataFrame: Preprocessed (or decoded) DataFrame

    arr = np.load(npz_path, allow_pickle=True)
    data = arr["data"]
    columns = arr["columns"]
    df = pd.DataFrame(data, columns=columns)

    if decode and "encodings" in arr:
        encoding_metadata = arr["encodings"].item()

        for col, meta in encoding_metadata.items():
            if meta["type"] == "label":
                # Reverse label encoding
                classes = meta["classes"]
                df[col] = undo_label_encoding(df[col].astype(int).tolist(), classes)

            elif meta["type"] == "one_hot":
                # Reverse one-hot encoding
                one_hot_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                one_hot_encoded = df[one_hot_cols].to_numpy()
                labels = undo_one_hot_encoding(one_hot_encoded, meta["classes"])

                # Replace one-hot columns with decoded column
                df = df.drop(columns=one_hot_cols)
                df[col] = labels

    return df


# ================= IMAGE =================
def load_preprocessed_image(npz_path):
    # Load preprocessed single image
    arr = np.load(npz_path, allow_pickle=True)
    return arr["image"]  # shape (224, 224, 3)


# ================= VIDEO =================
def load_preprocessed_video(npz_path):
    # Load preprocessed video frames
    arr = np.load(npz_path, allow_pickle=True)
    return arr["frames"]  # shape (#frames, 224, 224, 3)


# ================= AUDIO =================
def load_preprocessed_audio(npz_path):
    # Load preprocessed audio waveform + MFCC features
    arr = np.load(npz_path, allow_pickle=True)
    waveform = arr["waveform"]
    sr = int(arr["sr"])
    mfcc = arr["mfcc"]
    return {"waveform": waveform, "sr": sr, "mfcc": mfcc}


# ================= SIGNAL =================
def load_preprocessed_signal(npz_path):
    # Load preprocessed biosignal (EDF) with headers
    arr = np.load(npz_path, allow_pickle=True)
    signals = arr["signals"]
    signal_headers = arr["signal_headers"].tolist()
    header = arr["header"].item()
    return {"signals": signals, "signal_headers": signal_headers, "header": header}



