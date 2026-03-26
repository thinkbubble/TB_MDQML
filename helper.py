# helper.py just contains a bunch of functions
# that can be used in any program, they are program
# and/or environment agnostic. They may need 


from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pandas as pd
import numpy as np
import random
import string
import pytz
import json
import math
import os


def generate_unique_id(length=10):
        characters = string.ascii_letters + string.digits # All letters (lowercase + uppercase) and digits
        unique_id = ''.join(random.choices(characters, k=length))
        return unique_id


def identify_that_file_types_in_folder_are_allowed(folder_name):
    """
    Biosignals
    EDF (European Data Format) EDF is a good choice for biosignals due to its ability to handle multiple channels and its widespread use in the medical field.
    MAT and HDF5 are preferred if you're working with MATLAB or need to handle large datasets efficiently with rich metadata.
    CSV is easy to use and understand but might not be suitable for large datasets or complex biosignals.
    WAV might be used if the biosignal is naturally in an audio format, like phonocardiograms (heart sounds).
    """

    """
    Look more into these data (file) types:
    BIN, OBJ, STL, PLY, STEP/IGES, MP4, AVI, MKV, MOV, DICOM, PCAP, SHP, GeoJSON, KML, NetCDF, C3D, FASTA, BAM, VCF
    """

    # Remove mp3 later and only use wav files for audio
    # Removed = 'mat', 'hdf5'
    allowed_extensions = {'edf', 'wav', 'mp3', 'mp4', 'png', 'jpg', 'csv', 'txt', 'gif'}
    extension_count = {}

    # Check all files in the specified folder
    try:
        for root, _, files in os.walk(folder_name):
            for filename in files:
                if filename.startswith('.'):
                    continue
                ext = os.path.splitext(filename)[1][1:].lower()
                
                # Count each extension
                if ext:
                    extension_count[ext] = extension_count.get(ext, 0) + 1

    except FileNotFoundError:
        return (False, f"Folder '{folder_name}' does not exist")

    # Find unallowed file types
    unallowed_extensions = {ext for ext in extension_count if ext not in allowed_extensions}
    if unallowed_extensions:
        return (False, f"Unallowed file types detected: {', '.join(unallowed_extensions)}. Allowed types are: mp3, mp4, png, jpg, csv, txt.")
    
    # Filter the count dictionary to only include allowed extensions
    allowed_extension_count = {ext: count for ext, count in extension_count.items() if ext in allowed_extensions}
    
    return (True, allowed_extension_count)


def gather_folder_statistics(folder_name):

    file_types = identify_that_file_types_in_folder_are_allowed(folder_name)

    print('file_types: ', file_types)

    #if number_of_files == 0:
        #return 0, {}, 0, 'null', 'null', {}
    
    number_of_files = count_number_of_files_in_folder(folder_name)
    print("number_of_files: ", number_of_files)

    
    total_file_size_in_bytes = calculate_total_file_size_in_folder(folder_name)
    print("total_file_size_in_bytes: ", total_file_size_in_bytes)

    rows_and_columns = calculate_total_rows_and_columns(folder_name)
    if isinstance(rows_and_columns, tuple):
        total_rows, total_columns = rows_and_columns
    else:
        total_rows, total_columns = 'null', 'null'

    print('number_of_files: ', number_of_files)
    print('total_file_size_in_bytes: ', total_file_size_in_bytes)
    print('total_rows: ', total_rows)
    print('total_columns: ', total_columns)

    return number_of_files, file_types, total_file_size_in_bytes, total_rows, total_columns


def count_number_of_files_in_folder(folder_name):
    """Counts the total number of visible files in the specified folder, excluding hidden files."""
    total_files = 0
    try:
        for root, _, files in os.walk(folder_name):
            for filename in files:
                if filename.startswith('.'):
                    continue
                total_files += 1
    except FileNotFoundError:
        return f"Folder '{folder_name}' does not exist"
    return total_files


def calculate_total_file_size_in_folder(folder_name):
    """Calculates the total file size of all files in the folder in bytes."""
    total_size = 0
    try:
        for root, _, files in os.walk(folder_name):
            for filename in files:
                if filename.startswith('.'):
                    continue
                file_path = os.path.join(root, filename)
                total_size += os.path.getsize(file_path)
    except FileNotFoundError:
        return f"Folder '{folder_name}' does not exist"
    return total_size


def calculate_total_rows_and_columns(folder_name):
    total_rows = 0
    unique_columns = set()

    try:
        for root, _, files in os.walk(folder_name):
            for filename in files:
                if filename.startswith('.'):
                    continue

                if filename.lower().endswith('.csv'):
                    file_path = os.path.join(root, filename)
                    df = pd.read_csv(file_path)
                    total_rows += len(df)
                    unique_columns.update(df.columns)
        # List all files in the directory
        #for entry in os.scandir(folder_name):
            # Check if the file is a CSV
            #if entry.is_file() and entry.name.endswith('.csv'):
                # Read the CSV file
                #df = pd.read_csv(entry.path)
                # Add the number of rows in this file to the total count
                #total_rows += len(df)
                # Add column names to the set of unique columns
                #unique_columns.update(df.columns)

    except FileNotFoundError:
        return f"Folder '{folder_name}' does not exist"
    except Exception as e:
        return f"An error occurred: {str(e)}"

    # Return the total number of rows and the number of unique columns
    return total_rows, len(unique_columns)


def append_this_job_data_to_csv(unique_id, folder_length, file_counts, folder_bytes, number_of_rows, number_of_columns, file_path='job_data.csv'):
    """
    Appends job data to a CSV file. Logs existence and counts of specific file types in a folder.

    Args:
    unique_id (str): Unique identifier for the job.
    goal (str): Goal or purpose of the job.
    folder_length (int): Total number of files in the folder.
    file_counts (dict): Dictionary with file counts keyed by file type.
    folder_bytes (int): Total size of the folder in bytes.
    number_of_rows (int): Number of rows processed (if applicable).
    number_of_columns (int): Number of columns processed (if applicable).
    file_path (str): Path to the CSV file for logging.

    Returns:
    None
    """
    # Define the CSV header structure
    headers = [
        'unique_id', 'folder_length', 
        'mp3', 'mp3_count', 'mp4', 'mp4_count', 
        'png', 'png_count', 'jpg', 'jpg_count', 
        'csv', 'csv_count', 'txt', 'txt_count', 
        'wav', 'wav_count', 'edf', 'edf_count', 
        'gif', 'gif_count',
        'folder_bytes', 'number_of_rows', 'number_of_columns'
    ]

    # Prepare data dictionary
    data = {header: 0 for header in headers}  # Initialize all to zero or appropriate defaults
    data.update({
        'unique_id': unique_id,
        'folder_length': folder_length,
        'folder_bytes': folder_bytes,
        'number_of_rows': number_of_rows,
        'number_of_columns': number_of_columns
    })

    # File types to check
    file_types = ['mp3', 'mp4', 'png', 'jpg', 'csv', 'txt', 'wav', 'edf', 'gif']
    
    # Update data dictionary with file counts and existence
    for file_type in file_types:
        data[f"{file_type}"] = 1 if file_type in file_counts else 0
        data[f"{file_type}_count"] = file_counts.get(file_type, 0)

    # Convert data dictionary to DataFrame
    df_new_row = pd.DataFrame([data], columns=headers)

    # Check if file exists
    if os.path.exists(file_path):
        try:
            # Check if file is empty
            if os.stat(file_path).st_size == 0:
                # File is empty, write headers and data
                df_new_row.to_csv(file_path, index=False)
            else:
                # File is not empty, append data
                df = pd.read_csv(file_path)
                df = pd.concat([df, df_new_row], ignore_index=True)
                df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error reading or writing to {file_path}: {e}")
    else:
        # File does not exist, create it
        df_new_row.to_csv(file_path, index=False)


def random_choice(min_val, max_val, log_scale=False):
    # Check if both min_val and max_val are integers
    if isinstance(min_val, int) and isinstance(max_val, int):
        # Return a random integer between min_val and max_val (inclusive)
        return random.randint(min_val, max_val)
    else:
        if log_scale:
            # Apply logarithmic scale if log_scale is True
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            random_log = random.uniform(log_min, log_max)
            #return 10 ** random.uniform(-3, 3)
            return round(10**random_log, 3)  # Convert back from log scale and round to three decimal places
        # Return a random float between min_val and max_val (inclusive)
        return round(random.uniform(min_val, max_val), 2)



#x = random_choice(0.001, 1, True)
#print(x)

def read_from_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        # Now 'data' is a Python dictionary
        return data


def write_to_json(file_name, sample):
    with open(file_name, 'w') as fp:
        json.dump(sample, fp, indent=4)


"""
ai_tasks = read_from_json('tasks.json')
tasks = list(ai_tasks.keys())
new_dict = {}
for t in range(0, len(tasks)):
    new_dict[tasks[t]] = t
write_to_json('goal_dict.json', new_dict)
#print(list(ai_tasks.keys()))

ai_tasks = read_from_json('models.json')
tasks = list(ai_tasks.keys())
print(len(tasks))

"""


# $$$$$   PREPROCESSING HELPER    $$$$$




# Converts text in columns to numbers - only works for low dimensionality (Red, Blue, Orange, etc.)
def label_encoding(list_of_labels):

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(list_of_labels)

    #print(encoded_labels)  # Output: [0, 2, 1, 2, 0]
    #print(list(label_encoder.classes_))  # Output: ['Blue', 'Red', 'Yellow']
    #print([str(item) for item in label_encoder.classes_])
    return encoded_labels, [str(item) for item in label_encoder.classes_]

#colors = ['Blue', 'Red', 'Yellow', 'Red', 'Blue']
#encoded_labels, text_classes = label_encoding(colors)
#print(encoded_labels)
#print(text_classes)

# Converts the encoded labels back to text
def undo_label_encoding(encoded_labels, text_classes):

    # Create a dictionary that maps encoded labels back to text labels
    label_to_text = {index: label for index, label in enumerate(text_classes)}

    # Map each encoded label back to its corresponding text label
    original_labels = [label_to_text[label] for label in encoded_labels]

    return original_labels

#decoded_labels = undo_label_encoding(encoded_labels, text_classes)
#print(decoded_labels)

# Function to apply one-hot encoding
def one_hot_encoding(list_of_labels):
    """
    One-hot encodes a list of labels.

    Args:
    list_of_labels (list): The categorical labels to one-hot encode.

    Returns:
    tuple: A numpy array of one-hot encoded labels and the list of unique classes.
    """
    unique_labels = sorted(set(list_of_labels))  # Get unique labels and sort them for consistency
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Create label to index mapping

    # Create the one-hot encoded matrix
    one_hot_encoded = np.zeros((len(list_of_labels), len(unique_labels)))

    for i, label in enumerate(list_of_labels):
        one_hot_encoded[i, label_to_index[label]] = 1  # Set the corresponding column to 1
    
    return one_hot_encoded, unique_labels

# Function to undo one-hot encoding
def undo_one_hot_encoding(one_hot_encoded, unique_labels):
    """
    Reverts one-hot encoded data back to the original categorical labels.

    Args:
    one_hot_encoded (np.array): The one-hot encoded matrix.
    unique_labels (list): The original list of unique classes (labels).

    Returns:
    list: The original categorical labels.
    """
    index_to_label = {idx: label for idx, label in enumerate(unique_labels)}  # Create index to label mapping

    # Get the index of the max value (1) in each row to determine the original label
    original_labels = [index_to_label[np.argmax(row)] for row in one_hot_encoded]

    return original_labels



### $$$ Type Checking $$$


# Boolean
def is_bool(var):
    return isinstance(var, bool)

# Integer
def is_integer(var):
    return isinstance(var, int)

# Float
def is_float(var):
    return isinstance(var, float)

# String
def is_string(var):
    return isinstance(var, str)

# nan
def is_it_nan(value):

    #Check if the provided value is NaN (Not a Number).
    try:
        return np.isnan(value)
    except TypeError:
        return False  # Return False for data types that cannot be NaN

# List
def is_list(var):
    return isinstance(var, list)

# Tuple
def is_tuple(var):
    return isinstance(var, tuple)

# Dictionary
def is_dict(var):
    return isinstance(var, dict)


# List of tuples with int and tuple of ints
def is_list_of_tuples_int_and_tuple_of_ints(data):

    #Check if data is a list of tuples, where each tuple consists of an integer and a tuple of integers.
    
    if isinstance(data, list):
        return all(isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], int) and isinstance(t[1], tuple) and all(isinstance(i, int) for i in t[1]) for t in data)
    return False


def is_list_of_tuples_matching_types(data):

    #Check if data is a list of tuples, with all tuples containing elements of the same type and all tuples being of the same structure.
    
    if not isinstance(data, list) or not all(isinstance(t, tuple) for t in data):
        return False

    # Check if all tuples are of the same length and types match across each position in the tuples
    if len(set(len(t) for t in data)) != 1:
        return False  # All tuples must be the same length

    # Ensure all elements in each tuple position are of the same type across the list
    for idx in range(len(data[0])):
        expected_type = type(data[0][idx])
        if not all(isinstance(t[idx], expected_type) for t in data):
            return False

    return True


def is_list_of_tuples_ints_or_floats(data):

    #Check if data is a list of tuples, where each tuple consists entirely of integers or entirely of floats,
    #and the entire list must consist of either all tuples of integers or all tuples of floats.

    if not isinstance(data, list) or not all(isinstance(t, tuple) for t in data):
        return False

    # Check if the first element determines the type for the entire list
    if not data:  # Check if the list is empty
        return True  # An empty list technically fulfills the condition

    # Determine if the first tuple consists of integers or floats
    first_tuple_type = int if all(isinstance(x, int) for x in data[0]) else float if all(isinstance(x, float) for x in data[0]) else None

    if first_tuple_type is None:
        return False  # First tuple is neither all int nor all float

    # Ensure all tuples match the type of the first tuple
    return all(all(isinstance(x, first_tuple_type) for x in t) for t in data)



# Pandas DataFrame
def is_pd_dataframe(var):
    return isinstance(var, pd.DataFrame)

# Numpy Array
def is_np_array(var):
    return isinstance(var, np.ndarray)

# List of Numpy Arrays
def is_list_of_np_arrays(var):
    return isinstance(var, list) and all(isinstance(x, np.ndarray) for x in var)

# 2D Numpy Array
def is_2d_np_array(var):
    return isinstance(var, np.ndarray) and var.ndim == 2


# List or None - Optional
def is_list_optional(var):
    return var is None or isinstance(var, list)



# List of Integers
def is_list_of_integers(var):
    return isinstance(var, list) and all(isinstance(x, int) for x in var)

# Integer or None
def is_int_or_none(var):
    return var is None or isinstance(var, int)

# List of Tuples - Matching Types
def is_list_of_tuples_matching_types(var):
    if not isinstance(var, list) or not all(isinstance(x, tuple) for x in var):
        return False
    return all(type(x) == type(var[0]) for x in var[1:])



# Integer or 'auto'
def is_int_or_auto(var):
    return isinstance(var, int) or var == 'auto'

# Tuple of Integers
def is_tuple_of_ints(var):
    return isinstance(var, tuple) and all(isinstance(x, int) for x in var)

def get_current_timestamp_new_york():
    time_zone_new_york = pytz.timezone('America/New_York')
    now = datetime.now(time_zone_new_york)
    timestamp = int(round(now.timestamp()))
    timestamp = timestamp
    return timestamp


def build_master_dictionary(txt_data, csv_data, image_data, video_data, audio_data, signal_data):
    # Build a master dictionary grouped by data type
    master_data = {
        "txt": txt_data if txt_data else {},
        "csv": csv_data if csv_data else {},
        "image": image_data if image_data else {},
        "video": video_data if video_data else {},
        "audio": audio_data if audio_data else {},
        "signal": signal_data if signal_data else {}
    }

    print("\n=== Master Preprocessing Results ===")
    for modality, data in master_data.items():
        print(f"{modality.upper()}:\n {data}\n")
    print("====================================\n")

    return master_data