import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point,Polygon

street_map = gpd.read_file('/home/shailesh/RC/Data/bengaluru_india.geojson')
