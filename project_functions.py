
from _1_sensing_and_signal_ingestion import sensing_and_signal_ingestion

def main():

    MASTER_UNIQUE_ID, external_data_mapping, internal_data = sensing_and_signal_ingestion()

    print('MASTER_UNIQUE_ID: ', MASTER_UNIQUE_ID)

    print('external_data_mapping: ', external_data_mapping)

    print('internal_data: ', internal_data)


    # Begin your coding here. Previous things may not work, but
    # most likely your data is preprocessed correctly at this point.