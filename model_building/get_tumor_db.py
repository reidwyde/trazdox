import platform
import os
import pandas as pd

def get_tumor_db():
    
    if platform.system() == 'Windows':  # check the file system formatting
        filtered_filename = os.getcwd() + '\..\data\\filtered_data_csv.csv'
    else:
        filtered_filename = os.getcwd() + '/../data//filtered_data_csv.csv'
        #!MKL_THREADING_LAYER=GNU
    tumor_size_db = pd.read_csv(filtered_filename)
    return tumor_size_db