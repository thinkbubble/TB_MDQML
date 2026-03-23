
from preprocess_data import sensing_and_signal_ingestion

def main():

    # AFTER YOU INGEST THE SAME DATA ONCE, you can switch to
    # using load_data.py to load from cleaned_data folder.
    # ie.
    # video_data = load_preprocessed_video(./cleaned_data/npz_path)
    
    MASTER_UNIQUE_ID, external_data_mapping, internal_data = sensing_and_signal_ingestion()

    print('MASTER_UNIQUE_ID: ', MASTER_UNIQUE_ID)

    print('external_data_mapping: ', external_data_mapping)

    print('internal_data: ', internal_data)


    # Begin your coding here. Previous things may not work, but
    # most likely your data is preprocessed correctly at this point.
    # This is where you will construct your entire pipeline. 
    # But all the functions you create should be in project_functions.py
    # or in new_helper.py, revisions made to load_data.py or _1_sensing_and_signals

    """
    
    Future Steps

    Feature Engineering / Representation
    Train / Validation / Test Split
    Model Selection (Use multiple for each goal and dataset - must compare)
    Training 
    Evaluation (Metrics) (Log all results - needs to be automated)
    Iteration / Tuning
    Deployment / Inference

    """

main()