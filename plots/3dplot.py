import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# approximate radius of earth in km
R = 6373.0

data = pd.read_csv('rc_latlong.txt', sep='\t')
data = data.sort_values(['latitude','longitude'])
print(data)

lat = list(data['latitude'])
lon = list(data['longitude'])
alt = list(data['altitude (m)'])

dis = [0]
for i in range(1,len(lat)):
    lat1 = radians(lat[i-1])
    lon1 = radians(lon[i-1])
    lat2 = radians(lat[i])
    lon2 = radians(lon[i])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dis.append(R * c)

print(dis)

fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.scatter(lat, lon, alt, c='r', marker='o')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Altitude (m)')

ax = fig.add_subplot(212)
ax.scatter(dis, lon, alt, c='r', marker='o')
ax.set_xlabel('Distance')
ax.set_ylabel('Altitude (m)')

plt.show()