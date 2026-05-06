import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from preprocess_data import sensing_and_signal_ingestion

def run_test():

    # AFTER YOU INGEST THE SAME DATA ONCE, you can switch to
    # using load_data.py to load from cleaned_data folder.
    # ie.
    # video_data = load_preprocessed_video(./cleaned_data/npz_path)
    my_folder = 'team_samuel'
    raw_data_folder_path = f'./{my_folder}/Data/raw_data'
    cleaned_data_folder_path = f'./{my_folder}/Data/cleaned_data'
    checkpoint_file_path = f'./{my_folder}/Data/checkpoint.txt'

    
    print('raw_data_folder_path: ', raw_data_folder_path)

    print('cleaned_data_folder_path: ', cleaned_data_folder_path)

    print('checkpoint_file_path: ', checkpoint_file_path)


    
    MASTER_UNIQUE_ID, external_data_mapping, internal_data = sensing_and_signal_ingestion(raw_data_folder_path, cleaned_data_folder_path, checkpoint_file_path)

    print('MASTER_UNIQUE_ID: ', MASTER_UNIQUE_ID)

    print('external_data_mapping: ', external_data_mapping)

    print('internal_data: ', internal_data)



run_test()