#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:55:22 2019

@author: shailesh
"""

import gmplot
import pandas as pd
import json
import numpy as np

with open('/home/shailesh/RC/Data/Uber_Movement/bangalore_wards.json') as f:
	data = json.load(f)
	data2 = pd.read_csv("/home/shailesh/RC/Data/Uber_Movement/Monthly/bangalore-wards-2019-2-OnlyWeekdays-MonthlyAggregate.csv")


def findWardCentroid(d):
	properties = d['properties'] 
	wardName = properties['WARD_NAME']
	movementID = properties['MOVEMENT_ID']
	vertices = np.array(d['geometry']['coordinates'])[0][0]
	centroid = np.mean(vertices,axis = 0)
	centroid = centroid[::-1]
	return (wardName,movementID,centroid)

def returnCentroidList(dictionary):
	return [findWardCentroid(d) for d in dictionary['features']]


BangaloreMap = gmplot.GoogleMapPlotter(12.9716,77.5946,13)

wardList = returnCentroidList(data)

for ward in wardList:
	wardName = ward[0]
	wardID = ward[1]
	lat, long = list(ward[2])
	BangaloreMap.marker(lat,long, title=wardName+', '+wardID)
	
sensorList = [('HSR Layout (sector-4)',[12.914647,77.638493]),('IISC - Campus',[13.017016,77.570733]),('Sarjapura (Rainbow Drive)',
[12.907983,77.687196]),('Public Affairs Center, Jigani - Bommasandra Link Road',[12.799788,77.661561])]

for sensor in sensorList:
	lat,long = sensor[1]
	BangaloreMap.marker(lat,long,title=sensor[0],color = '#4B0082')
BangaloreMap.apikey = 'AIzaSyC3PGO0YO_ScHdVP9jj0b7T4ahljEii2qA'
BangaloreMap.draw("./Bangalore2.html")

#gmap1 = gmplot.GoogleMapPlotter(30.3164945, 
#                                78.03219179999999, 13 )

#gmap1.apikey = 'AIzaSyC3PGO0YO_ScHdVP9jj0b7T4ahljEii2qA'
#gmap1.draw( "./firstMap.html" )






#latitudes
#
#12.914647,77.638493
	

	