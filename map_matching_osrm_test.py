##pc0179 on the Kboard
# quick osrm map-matching investigation


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
#import matplotlib.pyplot as plt

#for Line-of-Sight Model
#from shapely.geometry import Point, LineString, shape, mapping, MultiLineString
#import fiona
#import geopandas as gp
#from geopandas.tools import sjoin





#1. Query taxi trace database for trace data at a particular time, T

#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)

# probably start the main loop here... for seconds in day...
#fur mike-pc
#execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude,x,y FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-t_margin),str(T+t_margin)))

#Time of interest:
start_time = dt.datetime(2014,2,3,6,0,15)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*13),30)) #search for 14 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 15 #i.e. thirty second chunks
t_accept = 1 #second either side? just use this value for position

num_real_connections = []
num_unreal_connections = []
total_conns = []
D_min = 5 # 15 #minimum distance in metres actually worth interpolating...


execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 

taxidf = pdsql.read_sql_query(execution_str,connection)
taxidf = taxidf.drop_duplicates() #removes duplicates, incase....
#snap co-ordinates to nearest road segment, no intelligence, no map-matching, just snapping.
#taxidf = Snap2Road(taxidf)





