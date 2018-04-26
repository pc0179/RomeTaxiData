#nearest neighbours...

import psycopg2
import pandas.io.sql as pdsql
import pandas as pd
from sqlalchemy import create_engine
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import numpy as np
import osrm

import time
import datetime as dt

# now on NiGB

#NOTE: [year,month,day,hour,min,second]
d = dt.datetime(2014,2,11,6,0,0)
t_unix = int(time.mktime(d.timetuple()))
t_range = 2 # +/-in seconds (ie 30s either side = 1 minute of search activity)

connect_str = "dbname='matchedtaxitraces' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)


execution_str = ("SELECT taxi_id, unix_ts, mlatitude, mlongitude FROM matchedtaxidata WHERE day_num = %s" % str(d.day-1))
#execution_str = ("SELECT taxi_id,unix_ts,mlatitude,mlongitude FROM matchedtaxidata WHERE unix_ts BETWEEN %s AND %s" % (str(t_unix-t_range),str(t_unix+t_range)))


taxidf = pdsql.read_sql_query(execution_str,connection)

within_range_taxidf = taxidf[taxidf['unix_ts'].between(int(t_unix-t_range), int(t_unix+t_range), inclusive=True)]

gps_locs = within_range_taxidf[['mlatitude','mlongitude']]
gps_locs4table = [list(x) for x in gps_locs.values]

list_ids = list(gps_locs.index)
list_ids2 = []
for i in range(0,len(list_ids)):
	list_ids2.append(str(list_ids[i]))


time_matrix, snapped_coords = osrm.table(gps_locs4table, ids_origin=list_ids2, output='dataframe')

