#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:02:17 2019

@author: shailesh
"""

import numpy as np
import keras
import pandas as pd
import glob
import statistics
from keras.models import Sequential
from keras.layers import Dense
from fancyimpute import IterativeImputer as MICE
import matplotlib.pyplot as plt

path = ('../Data/Uber_Movement/Daily_final/')
all_files = glob.glob(path + "*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame_uber = pd.concat(li, axis=0, ignore_index=True)

frame_uber['Date'] = pd.to_datetime(frame_uber['Date'])
frame_uber.sort_values(by=['Date'])

means = []
for index, row in frame_uber.groupby(frame_uber.Date.dt.date):
    m = []
    m.append((statistics.mean(list(row['Early Morning Mean Travel Time (Seconds)'])), 1))
    m.append((statistics.mean(list(row['AM Mean Travel Time (Seconds)'])), 2))
    m.append((statistics.mean(list(row['Midday Mean Travel Time (Seconds)'])), 3))
    m.append((statistics.mean(list(row['PM Mean Travel Time (Seconds)'])), 4))
    m.append((statistics.mean(list(row['Evening Mean Travel Time (Seconds)'])), 5))
    means.append(m)

path = ('../Data/Precipitation/')
# all_files = glob.glob(path + "*.csv")
all_files = ['../Data/Precipitation\\yuktix1_download-April.csv', '../Data/Precipitation\\yuktix1_download-May.csv', '../Data/Precipitation\\yuktix1_download-June.csv']

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame_prec = pd.concat(li, axis=0, ignore_index=True)

for index, row in frame_prec.iterrows():
    x = row['Date'].split()
    x[0] += '-2019'
    frame_prec.at[index, 'Date'] = x[0] + ' ' + x[1] + ' '+ x[2]

frame_prec['Date'] = pd.to_datetime(frame_prec['Date'])
frame_prec.sort_values(by=['Date'])

rains = []
dates = []
for idx, rows in frame_prec.groupby(frame_prec.Date.dt.date):
    rain = []
    r = rows.resample('H', on='Date').mean()
    dates.append(r.index.values)
    try:
        rain.append(statistics.mean(list(r.between_time('00:00:01', '07:00:00', include_start=True, include_end=False)['Rain'])))
    except:
        rain.append(np.nan)
    try:
        rain.append(statistics.mean(list(r.between_time('07:00:00', '10:00:00', include_start=True, include_end=False)['Rain'])))
    except:
        rain.append(np.nan)
    try:
        rain.append(statistics.mean(list(r.between_time('10:00:00', '16:00:00', include_start=True, include_end=False)['Rain'])))
    except:
        rain.append(np.nan)
    try:    
        rain.append(statistics.mean(list(r.between_time('16:00:00', '19:00:00', include_start=True, include_end=False)['Rain'])))
    except:
        rain.append(np.nan)
    try:
        rain.append(statistics.mean(list(r.between_time('19:00:00', '23:59:59', include_start=True, include_end=False)['Rain'])))
    except:
        rain.append(np.nan)
    rains.append(rain)

rains_corr = []
for i in rains:
    for j in i:
        rains_corr.append(j)
means_corr = []
for i in means:
    for j in i:
        means_corr.append(j)

rcorr = rains_corr
mcorr = []
time_segs = []
for i in means_corr:
    mcorr.append(i[0])
    time_segs.append(i[1])
print(len(rcorr))
print(len(mcorr))

mice_imputed = MICE().fit_transform(pd.DataFrame(list(zip(mcorr, rcorr)), columns=['Traffic', 'Rain']))
rains_corr = mice_imputed[:,1]
mcorr = mice_imputed[:,0]
means_corr = []
for i in range(len(mcorr)):
    means_corr.append((mcorr[i], time_segs[i]))

traffic_data = []
rain_data = []
traffic_data_next = []

for i in range(len(rains_corr)-1):
    if rains_corr[i] != 0:
        traffic_data.append(means_corr[i])
        rain_data.append(rains_corr[i])
        traffic_data_next.append(means_corr[i+1])

# print(traffic_data_next)
# print(len(traffic_data_next))
# print(traffic_data)
# print(len(traffic_data))

traffic_rain_temp = [(traffic_data[i], rain_data[i]) for i in range(0, len(traffic_data))]
traffic_rain = []
for i in range(len(traffic_rain_temp)):
    traffic_rain.append((traffic_rain_temp[i][0][0], traffic_rain_temp[i][1], traffic_rain_temp[i][0][1]))

morning_traffic_norain = []
ampeak_traffic_norain = []
midday_traffic_norain = []
pmpeak_traffic_norain = []
evening_traffic_norain = []
j = 0
for i in rains:
    if i[0] == 0:
        morning_traffic_norain.append(means[j][0][0])
    if i[1] == 0:
        ampeak_traffic_norain.append(means[j][1][0])
    if i[2] == 0:
        midday_traffic_norain.append(means[j][2][0])
    if i[3] == 0:
        pmpeak_traffic_norain.append(means[j][3][0])
    if i[4] == 0:
        evening_traffic_norain.append(means[j][4][0])
    j += 1

# print(traffic_data_next)

labels = []
for i in range(len(traffic_data_next)):
    # early morning
    if traffic_data_next[i][1] == 1:
        if ((traffic_data_next[i][0] - statistics.mean(morning_traffic_norain))/statistics.mean(morning_traffic_norain) >= 0.25):
            labels.append([0,0,1])
        elif ((traffic_data_next[i][0] - statistics.mean(morning_traffic_norain))/statistics.mean(morning_traffic_norain) >= 0.10):
            labels.append([0,1,0])
        else:
            labels.append([1,0,0])
        print((traffic_data_next[i][0] - statistics.mean(morning_traffic_norain))/statistics.mean(morning_traffic_norain))
    # am peak
    elif traffic_data_next[i][1] == 2:
        if ((traffic_data_next[i][0] - statistics.mean(ampeak_traffic_norain))/statistics.mean(ampeak_traffic_norain) >= 0.25):
            labels.append([0,0,1])
        elif ((traffic_data_next[i][0] - statistics.mean(ampeak_traffic_norain))/statistics.mean(ampeak_traffic_norain) >= 0.10):
            labels.append([0,1,0])
        else:
            labels.append([1,0,0])
        print((traffic_data_next[i][0] - statistics.mean(ampeak_traffic_norain))/statistics.mean(ampeak_traffic_norain))
    # midday
    elif traffic_data_next[i][1] == 3:
        if ((traffic_data_next[i][0] - statistics.mean(midday_traffic_norain))/statistics.mean(midday_traffic_norain) >= 0.25):
            labels.append(3)
        elif ((traffic_data_next[i][0] - statistics.mean(midday_traffic_norain))/statistics.mean(midday_traffic_norain) >= 0.10):
            labels.append([0,1,0])
        else:
            labels.append([1,0,0])
        print((traffic_data_next[i][0] - statistics.mean(midday_traffic_norain))/statistics.mean(midday_traffic_norain))
    # pm peak
    elif traffic_data_next[i][1] == 4:
        if ((traffic_data_next[i][0] - statistics.mean(pmpeak_traffic_norain))/statistics.mean(pmpeak_traffic_norain) >= 0.25):
            labels.append([0,0,1])
        elif ((traffic_data_next[i][0] - statistics.mean(pmpeak_traffic_norain))/statistics.mean(pmpeak_traffic_norain) >= 0.10):
            labels.append([0,1,0])
        else:
            labels.append([1,0,0])
        print((traffic_data_next[i][0] - statistics.mean(pmpeak_traffic_norain))/statistics.mean(pmpeak_traffic_norain))
    # evening
    elif traffic_data_next[i][1] == 5:
        if ((traffic_data_next[i][0] - statistics.mean(evening_traffic_norain))/statistics.mean(evening_traffic_norain) >= 0.25):
            labels.append([0,0,1])
        elif ((traffic_data_next[i][0] - statistics.mean(evening_traffic_norain))/statistics.mean(evening_traffic_norain) >= 0.10):
            labels.append([0,1,0])
        else:
            labels.append([1,0,0])
        print((traffic_data_next[i][0] - statistics.mean(evening_traffic_norain))/statistics.mean(evening_traffic_norain))
print(labels)

labels = np.asarray(labels)
traffic_rain = np.asarray(traffic_rain)
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=3))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(traffic_rain[:31], labels[:31], epochs=100)
model.summary()

_, accuracy = model.evaluate(traffic_rain[31:], labels[31:])
print('Accuracy: %.2f' % (accuracy*100))

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()