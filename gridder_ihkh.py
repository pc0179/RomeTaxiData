import RomeTaxiGlobalVars as RTGV
import psycopg2
import pandas.io.sql as pdsql
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import numpy as np





"""
actually not such a good idea, given what we really need to do
is first filter the data by at least removing all points outside of the 8*8km
add the other 'time_stamp... related tags, as columns, then
process the gps points and place them in 100m*100m blocks?


really the code below, should read straight from the raw data, 
imagine it as version 2.0 of the input pipeline to the database



or do it anyway, have wierd cols in numpy where they are abs(Datumlat-Lat)? then remove all points that are outside of the BoundingBox?
how to remove rows based on criteria, psql10!!!
booom = https://www.postgresql.org/docs/10/static/sql-delete.html #thats the fucking website!!!

that all said, maybe what's really important is working out average distance between points?
maybe check the damn average speed in the paper d=s*t


"""

KLARA_connect_str = "dbname='taxitraces' user='postgres' host='localhost' password='postgres'"
# quick reminder of available columns in database:
# cols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y']


KLARA_connection = psycopg2.connect(KLARA_connect_str)

KLARA_exec_str = "SELECT Lat1 Long1 FROM rometaxidata"


uniq_taxidf = pdsql.read_sql_query(KLARA_exec_str,KLARA_connection)


# ~~~~~ SOME MAGIC, GET THE LAT LONGS...

length_scale = 200 #metres? is this useful?
def Haversine_PC2(lon1,lat1,lon2,lat2):
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	dlon = lon2-lon1
	dlat = lat2-lat1
	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
	c = 2 * np.arcsin(np.sqrt(a))
	Hdistance = 6371e3*c  #working in metres!
	return Hdistance

def DatumLatLongToNPColRow(lat1,long1,length_scale):
	#where Datum is 'top left' coordinate of bounding box
	DatumLat = RTGV.DatumLat
	DatumLong = RTGV.DatumLong
	
	#for numpy Column Position
	ColN = Haversine_PC2(DatumLong, DatumLat, long1,DatumLat)
	coln = round(ColN,length_scale)
	#for numpy Row position
	RowM = Haversine_PC2(DatumLong,DatumLat, DatumLong, lat1)
	
	
	

