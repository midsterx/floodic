#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:02:17 2019

@author: shailesh
"""

import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(10,input_dim = 4, activation = 'sigmoid'))
model.add(Dense(3, activation = 'softmax'))


model.summary()
