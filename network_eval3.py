# pc0179 on the Kboard
# network_eval3.py , yup, third time's a charm!

#import matplotlib
#matplotlib.use('Agg')


import time
script_stat_time = time.time()

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

#for Line-of-Sight Model
from shapely.geometry import Point, LineString, shape, mapping, MultiLineString
import fiona
import geopandas as gp
from geopandas.tools import sjoin

#useful functions...
def ProcessMapMatchResults(matched_points, timestamps):

    #matched_longs = []
    #matched_lats = []
    nobody_index = []
    matched_ts = []
    matched_pos = []
    for i in range(0,len(matched_points['tracepoints'])):

        if matched_points['tracepoints'][i] is None:
            nobody_index.append(i)
        else:
            matched_ts.append(timestamps[i])
            #matched_longs.append(matched_points['tracepoints'][i]['location'][0])
            #matched_lats.append(matched_points['tracepoints'][i]['location'][1])
            matched_pos.append(tuple(matched_points['tracepoints'][i]['location'][0],matched_points['tracepoints'][i]['location'][1]))

    matchedf = pd.DataFrame({'mts':matched_ts,'mpos':matched_pos})
    return matchedf, nobody_index



#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)


#Time of interest:
#tuseday 4th feb 2014, 7am-7pm?
start_time = dt.datetime(2014,2,4,7,0,15)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*12),30)) #search for 12 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 15 #i.e. 30 second chunks
t_accept = 1 #second either side? just use this value for position

num_real_connections = []
num_unreal_connections = []
total_conns = []
D_min = 5 # 15 #minimum distance in metres actually worth interpolating...


# skeleton of main loop...
execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 

taxidf = pdsql.read_sql_query(execution_str,connection)
taxidf = taxidf.drop_duplicates() #removes duplicates, incase....



before_taxi_ids = list(beforedf.taxi_id.unique())
after_taxi_ids = list(afterdf.taxi_id.unique())
taxi_ids_2process = set(before_taxi_ids).intersection(after_taxi_ids)


#for each trace, another loop here...
#for taxi_id in taxi_ids_2process
#taxi_id = taxidf.taxi_id.unique()[1]

# map match.
taxi_subset = taxidf[taxidf.taxi_id==taxi_id].sort_values('unix_ts')
timestamps2match = taxi_subset.unix_ts.tolist()

taxi_pos2match = [tuple(x) for x in gps_subset[['longitude','latitude']].values]

matched_points = osrm.match(taxi_pos2match, overview="simplified", timestamps=timestamps2match, radius=None)

matchedf, nobody_index = ProcessMapMatchResults(matched_points, timestamps)

# remove 'lucky' points that are within t_accept range of T.
matchedf['ts_diff'] = matchedf.mts-T
# route&/interp

"""
post lunch, please be calm.
scenario 1
query 30s of data from dbass
drop duplicates
find taxis with trace data either side of T
for those with points either side of T
- map-match
- route/&interp
- estimate position at T



select taxi trace by ID
map match 30s of trace







