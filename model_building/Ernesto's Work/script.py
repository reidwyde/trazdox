import sys
import pandas as pd
data = pd.read_csv('error.txt',header=None,delimiter=' ')
orig_stdout = sys.stdout
f = open('error_m_sd.txt', 'w')
sys.stdout = f
for column in data:
    print data[column].mean(),data[column].std()
sys.stdout = orig_stdout
f.close()
