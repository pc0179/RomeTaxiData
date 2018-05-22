# hdawg on the Kboard.
# lets do some plotting, raw vs route vs match vs snap.
# investigation time = 1 hour.

#in general:
import numpy as np
import pandas as pd
import datetime as dt

#for routing:
import osrm
import polyline

#for querying taxi trace database:
import psycopg2
import pandas.io.sql as pdsql
from sqlalchemy import create_engine

#for plotting:
import matplotlib.pyplot as plt

#--------------------

# query/import raw taxi trace data
#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)
#Time of interest:
start_time = dt.datetime(2014,2,2,18,00,00)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*1),30)) #search for 14 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 60*60*1 # i.e. une heure #15 #i.e. thirty second chunks
t_accept = 1 #second either side? just use this value for position


T = Tstart_unix # for now....

execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 

taxidf = pdsql.read_sql_query(execution_str,connection)
taxidf = taxidf.drop_duplicates() #removes duplicates, incase....
#snap co-ordinates to nearest road segment, no intelligence, no map-matching, just snapping.

#for taxi_id in taxidf.taxi_id.unique():
taxi_id = taxidf.taxi_id.unique()[1]

# snapping to road segment
def Snap2Road(longitude,latitude):
#inputs: lists of longitudes and latidutes
#output: lists of snapped coords

    snapped_longitude = longitude
    snapped_latitude = latitude

    for i in range(0,len(snapped_latitude)):
        
        snapped_result = osrm.nearest((snapped_longitude[i],snapped_latitude[i]))
        snapped_longitude[i] = snapped_result['waypoints'][0]['location'][0]
        snapped_latitude[i] = snapped_result['waypoints'][0]['location'][1]
    
    return snapped_longitude, snapped_latitude

#a = [taxidf.latitude[taxidf.taxi_id==255].value,taxidf.longitude[taxidf.taxi_id==255].value]

snapped_long, snapped_lat = Snap2Road(taxidf.latitude[taxidf.taxi_id==taxi_id].tolist(),taxidf.longitude[taxidf.taxi_id==taxi_id].tolist())


# map-matching

#gps_subset = trace_data2match[['long1','lat1']]

gps_subset = taxidf[taxidf.taxi_id==taxi_id].sort_values('unix_ts')

gps_positions = [tuple(x) for x in gps_subset[['latitude','longitude']].values]
#mpmatched_points = osrm.match(gps_positions, overview="simplified", timestamps=gps_subset.unix_ts.tolist(), radius=None)

from polyline.codec import PolylineCodec
from polyline import encode as polyline_encode
atit = polyline.encode(gps_positions,6)

mpmatched_points = osrm.match(atit, overview="simplified", radius=None)
#mpmatched_points = osrm.match(atit, overview="simplified", timestamps=gps_subset.unix_ts.tolist(), radius=None)

#def MapMatch2Road():


"""
query, import say 1 hour
pick 1 taxi.
raw
map-match
snap
route(A-B)

plot above.
"""
