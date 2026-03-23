from helper import get_current_timestamp_new_york
from sklearn.preprocessing import StandardScaler
from helper import append_this_job_data_to_csv
from helper import gather_folder_statistics
from helper import build_master_dictionary
from sklearn.impute import SimpleImputer
from nltk.tokenize import word_tokenize
from helper import generate_unique_id
from helper import one_hot_encoding
from moviepy import VideoFileClip
from nltk.corpus import stopwords
from helper import label_encoding
from pyedflib import highlevel
import pandas as pd
import numpy as np
import librosa
import string
import nltk
import cv2
import os
import re


def setup_nltk_data():
    nltk_dir = './nltk_data'

    # Always tell nltk to look here, even if the folder already exists
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)

    # If folder already exists, skip download
    if os.path.exists(nltk_dir):
        return

    print(f"Creating '{nltk_dir}' and downloading required NLTK data...")
    os.makedirs(nltk_dir, exist_ok=True)

    # Download punkt and punkt_tab
    nltk.download('punkt', download_dir=nltk_dir)
    nltk.download('punkt_tab', download_dir=nltk_dir)
    nltk.download("stopwords", download_dir=nltk_dir)

# Run setup before using NLTK
setup_nltk_data()


def preprocess_txt_files_in_folder(folder_path, output_dir="./cleaned_data", 
                                   punctuation_choice=True, stop_words_choice=True):
    # Preprocess all .txt files by cleaning, normalizing, tokenizing,
    # and optionally removing punctuation/stopwords.
    # Saves each result as a compressed .npz file in cleaned_data.
    #
    # Args:
    #     folder_path (str): Path to the folder containing .txt files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #     punctuation_choice (bool): If False, punctuation will be removed.
    #     stop_words_choice (bool): If False, stopwords will be removed.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning txt pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                results[filename] = f"ERROR: Could not read {file_path}: {e}"
                continue

            # === Normalize ===
            text = text.lower()
            text = re.sub(r"\s+", " ", text).strip()

            # === Tokenize ===
            if not punctuation_choice:
                text = text.translate(str.maketrans("", "", string.punctuation))
            tokens = word_tokenize(text)

            # === Stopword removal ===
            if not stop_words_choice:
                sw = set(stopwords.words("english"))
                tokens = [word for word in tokens if word not in sw]

            # === Save as .npz ===
            save_path = os.path.join(output_dir, f"txt_{os.path.splitext(filename)[0]}_tokens.npz")
            try:
                np.savez_compressed(save_path, tokens=np.array(tokens, dtype="object"))
                results[filename] = save_path
            except Exception as e:
                results[filename] = f"ERROR: Could not save {save_path}: {e}"

    return results


def preprocess_csv_files_in_folder(folder_path, output_dir="./cleaned_data", threshold=10):
    # Preprocess all .csv files by normalizing numeric values, encoding
    # categorical columns with provided encoding helpers, handling dates,
    # and imputing missing values. Saves results to compressed .npz files.
    #
    # Args:
    #     folder_path (str): Path to the folder containing .csv files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #     threshold (int): Threshold for deciding between one-hot and label encoding.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning csv pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            try:
                df = pd.read_csv(file_path)
                encoding_metadata = {}  # to store mappings for undo

                # === Handle date columns ===
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["year"] = df["date"].dt.year
                    df["month"] = df["date"].dt.month
                    df["day"] = df["date"].dt.day
                    df = df.drop("date", axis=1)

                # === Handle missing values ===
                for col in df.columns:
                    if df[col].dtype in ["int64", "float64"]:
                        imputer = SimpleImputer(strategy="mean")
                    else:
                        imputer = SimpleImputer(strategy="most_frequent")
                    df[[col]] = imputer.fit_transform(df[[col]])

                # === Normalize numeric columns ===
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

                # === Encode categorical columns using helper functions ===
                categorical_cols = df.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    values = df[col].tolist()
                    num_unique = len(set(values))

                    if num_unique < threshold:
                        # One-hot encoding
                        one_hot_encoded, unique_labels = one_hot_encoding(values)
                        one_hot_df = pd.DataFrame(
                            one_hot_encoded,
                            columns=[f"{col}_{label}" for label in unique_labels]
                        )
                        df = df.drop(columns=[col]).reset_index(drop=True).join(one_hot_df)
                        encoding_metadata[col] = {"type": "one_hot", "classes": unique_labels}
                    else:
                        # Label encoding
                        encoded_labels, classes = label_encoding(values)
                        df[col] = encoded_labels
                        encoding_metadata[col] = {"type": "label", "classes": classes}

                # === Save as .npz ===
                save_path = os.path.join(output_dir, f"csv_{os.path.splitext(filename)[0]}_cleaned.npz")
                np.savez_compressed(
                    save_path,
                    data=df.to_numpy(),
                    columns=df.columns.to_numpy(),
                    encodings=np.array(encoding_metadata, dtype=object)  # save mappings
                )

                results[filename] = save_path

            except Exception as e:
                results[filename] = f"ERROR: Could not process {file_path}: {e}"

    return results



def preprocess_image_data_in_folder(folder_path, output_dir="./cleaned_data", 
                                    target_size=(224, 224), denoise=True):
    # Preprocess all image files in a folder by resizing, normalizing pixel values,
    # and optionally denoising. Saves results to compressed .npz files in cleaned_data.
    #
    # Args:
    #     folder_path (str): Path to the folder containing image files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #     target_size (tuple): Size to resize images to (width, height).
    #     denoise (bool): Whether to apply a denoising filter.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning image pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)

            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Image is empty or corrupted")

                # Resize
                image_resized = cv2.resize(image, target_size)

                # Convert to RGB
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

                # Normalize pixel values
                image_normalized = image_resized.astype("float32") / 255.0

                # Optional denoising
                if denoise:
                    image_normalized = cv2.medianBlur((image_normalized * 255).astype("uint8"), 3)
                    image_normalized = image_normalized.astype("float32") / 255.0

                # Save as compressed .npz
                save_path = os.path.join(output_dir, f"img_{os.path.splitext(filename)[0]}_cleaned.npz")
                np.savez_compressed(save_path, image=image_normalized)

                results[filename] = save_path

            except Exception as e:
                results[filename] = f"ERROR: Could not process {file_path}: {e}"

    return results


# $$$$$$$ VIDEO DATA FILE PRE-PROCESSING $$$$$$$
def preprocess_video_data_in_folder(folder_path, output_dir="./cleaned_data", target_size=(224, 224), fps=1):
    # Preprocess video files by extracting frames, resizing, normalizing.
    # Saves each video as a compressed .npz containing frames.
    #
    # Args:
    #     folder_path (str): Path to folder containing video files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #     target_size (tuple): Resize frames to (width, height).
    #     fps (int): Frames per second to sample from the video.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning video pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".mp4", ".mov", ".gif")):
            file_path = os.path.join(folder_path, filename)
            try:
                video = VideoFileClip(file_path)
                frames = []

                for frame in video.iter_frames(fps=fps, dtype="uint8"):
                    frame_resized = cv2.resize(frame, target_size)
                    frame_resized = frame_resized.astype("float32") / 255.0
                    frames.append(frame_resized)

                frames = np.stack(frames, axis=0) if frames else np.empty((0, *target_size, 3))

                save_path = os.path.join(output_dir, f"video_{os.path.splitext(filename)[0]}_cleaned.npz")
                np.savez_compressed(save_path, frames=frames)

                results[filename] = save_path

            except Exception as e:
                results[filename] = f"ERROR: Could not process {file_path}: {e}"

    return results


# $$$$$$$ AUDIO DATA FILE PRE-PROCESSING $$$$$$$
def preprocess_audio_data_in_folder(folder_path, output_dir="./cleaned_data", n_mfcc=13):
    # Preprocess audio files by loading waveform, normalizing, extracting MFCCs.
    # Saves each audio file as a compressed .npz with waveform + mfccs.
    #
    # Args:
    #     folder_path (str): Path to folder containing audio files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #     n_mfcc (int): Number of MFCC coefficients to extract.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning audio pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(folder_path, filename)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                audio = librosa.util.normalize(audio)  # normalize waveform
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

                save_path = os.path.join(output_dir, f"audio_{os.path.splitext(filename)[0]}_cleaned.npz")
                np.savez_compressed(save_path, waveform=audio, sr=sr, mfcc=mfcc)

                results[filename] = save_path

            except Exception as e:
                results[filename] = f"ERROR: Could not process {file_path}: {e}"

    return results


# $$$$$$$ SIGNAL DATA FILE PRE-PROCESSING $$$$$$$
def preprocess_signal_data_in_folder(folder_path, output_dir="./cleaned_data"):
    # Preprocess biosignal files (EDF) by reading signals and metadata.
    # Applies basic normalization and saves each as a compressed .npz.
    #
    # Args:
    #     folder_path (str): Path to folder containing .edf files.
    #     output_dir (str): Directory where preprocessed files will be stored.
    #
    # Returns:
    #     dict: {filename: saved_file_path}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\nBeginning signal pre-processing...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".edf"):
            file_path = os.path.join(folder_path, filename)
            try:
                signals, signal_headers, header = highlevel.read_edf(file_path)

                # Normalize each channel (z-score)
                signals = np.array(signals, dtype="float32")
                signals = (signals - signals.mean(axis=1, keepdims=True)) / (signals.std(axis=1, keepdims=True) + 1e-8)

                save_path = os.path.join(output_dir, f"edf_{os.path.splitext(filename)[0]}_cleaned.npz")
                np.savez_compressed(save_path, signals=signals, signal_headers=signal_headers, header=header)

                results[filename] = save_path

            except Exception as e:
                results[filename] = f"ERROR: Could not process {file_path}: {e}"

    return results



# Gather data from internal system to help guide decisions - Interoception
def gather_internal_data():

    pass


# Gather data from external world to help guide decisions
# Could add a sensor layer later that gathers data into a file
# and drops these files in raw_data folder
# Could make this a loop that continually checks that folder for new data
def gather_external_data(folder_name="./raw_data"):

    number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns = gather_folder_statistics(folder_name)

    if (number_of_files==False and file_types==False and total_file_size_in_bytes==False and total_rows==False and total_columns==False):
        return False, False, False, False, False, False, 'file_types_not_allowed'

    #goal = determine_and_encode_goal(goal, number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns)

    #if (goal[0] == 'GoalError'):
    #    return False, False, False, False, False, False, goal[0], goal[0]

    try:

        txt_data = preprocess_txt_files_in_folder(folder_name)
        csv_data = preprocess_csv_files_in_folder(folder_name)
        image_data = preprocess_image_data_in_folder(folder_name)
        video_data = preprocess_video_data_in_folder(folder_name)
        audio_data = preprocess_audio_data_in_folder(folder_name)
        signal_data = preprocess_signal_data_in_folder(folder_name)

        print("\n=== Preprocessing Results ===")
        print("TXT Data:\n", txt_data, "\n")
        print("CSV Data:\n", csv_data, "\n")
        print("Image Data:\n", image_data, "\n")
        print("Video Data:\n", video_data, "\n")
        print("Audio Data:\n", audio_data, "\n")
        print("Signal Data:\n", signal_data, "\n")
        print("================================\n")

        print("\n=== Preprocessing Summary ===")
        print(f"TXT files processed:     {len(txt_data) if txt_data else 0}")
        print(f"CSV files processed:     {len(csv_data) if csv_data else 0}")
        print(f"Image files processed:   {len(image_data) if image_data else 0}")
        print(f"Video files processed:   {len(video_data) if video_data else 0}")
        print(f"Audio files processed:   {len(audio_data) if audio_data else 0}")
        print(f"Signal files processed:  {len(signal_data) if signal_data else 0}")
        print("============================\n")

    except:

        return False, False, False, False, False, False, 'preprocessing_error'


    master_external_data = build_master_dictionary(
        txt_data, csv_data, image_data, video_data, audio_data, signal_data
    )

    return number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns, master_external_data
    #return txt_data, csv_data, image_data, video_data, audio_data, signal_data, goal[2], 'all_clear'

def log_first_checkpoint(number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns):

    unique_id = generate_unique_id(10)
    time_stamp = get_current_timestamp_new_york()

    if isinstance(file_types, tuple):
        file_types_dict = file_types[1]
    else:
        file_types_dict = file_types

    # Database Swap Later
    append_this_job_data_to_csv(unique_id, number_of_files, file_types[1], total_file_size_in_bytes, total_rows, total_columns, file_path='job_data.csv')
    #append_this_job_data_to_csv(unique_id, number_of_files, file_types[1], total_file_size_in_bytes, total_rows, total_columns, file_path='job_data.csv')
    # Start base entry
    first_checkpoint_entry = {
        "folder_length": number_of_files,
        "folder_bytes": total_file_size_in_bytes,
        "number_of_rows": total_rows,
        "number_of_columns": total_columns,
        "mp3_count": file_types_dict.get("mp3", 0),
        "mp4_count": file_types_dict.get("mp4", 0),
        "png_count": file_types_dict.get("png", 0),
        "jpg_count": file_types_dict.get("jpg", 0),
        "csv_count": file_types_dict.get("csv", 0),
        "txt_count": file_types_dict.get("txt", 0),
        "wav_count": file_types_dict.get("wav", 0),
        "edf_count": file_types_dict.get("edf", 0),
        "gif_count": file_types_dict.get("gif", 0),
        "time_stamp": time_stamp
    }

    print('fcpe: ', first_checkpoint_entry)
    
    """
    insert_result = omnexus_request('mongo', 'insert', {
        "database": "magi_db",
        "collection": "checkpoints",
        "data": first_checkpoint_entry
    })
    unique_id = insert_result.get("inserted_id") or insert_result.get("_id")
    """
    
    return unique_id


def sensing_and_signal_ingestion():

    internal_data = gather_internal_data()
    number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns, external_data_mapping = gather_external_data('./raw_data')
    
    MASTER_UNIQUE_ID = log_first_checkpoint(number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns)

    return MASTER_UNIQUE_ID, external_data_mapping, internal_data



