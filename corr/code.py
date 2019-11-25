import json
import csv
import pandas as pd
import glob
from numpy import cov
from scipy.stats import pearsonr, spearmanr


path = ('../../Data/Uber_Movement/Daily_final/')
all_files = glob.glob(path + "*.csv")

print(all_files)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame_uber = pd.concat(li, axis=0, ignore_index=True)
print(frame_uber.head)
# df_uber = frame_uber.loc[frame_uber['sourceid'].isin([160, 171])]
# df_final_uber = df_uber.loc[df_uber['dstid'].isin([160, 171])]

path = ('../../Data/Precipitation/')
all_files = glob.glob(path + "*.csv")

print(all_files)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame_prec = pd.concat(li, axis=0, ignore_index=True)

print(frame_prec.head)
# print(df_final_uber.head)

# covariance = cov(data1, data2)
# pcorr, _ = pearsonr(data1, data2)
# scorr, _ = spearmanr(data1, data2)


