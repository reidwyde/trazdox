import numpy as np

def parse_tumor_db(tumor_size_db):    
    ts = np.array(tumor_size_db['Day']).transpose() # dimension: (19,)
    Ts = np.array(tumor_size_db[['G1_avg','G2_avg','G3_avg','G4_avg','G5_avg','G6_avg']]).transpose() # indexing: group, time
    sigmas = np.array(tumor_size_db[['G1_sd','G2_sd','G3_sd','G4_sd','G5_sd','G6_sd']]).transpose()
    return ts, Ts, sigmas