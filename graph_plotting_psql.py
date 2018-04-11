import psycopg2
#import csv
import pandas.io.sql as pdsql
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import numpy as np

connect_str = "dbname='taxitraces' user='postgres' host='localhost' password='postgres'"
# quick reminder of available columns in database:
# cols = ['taxi_id','ts_dt','sim_t','sim_day_num','weekday_num','Lat1','Long1','x','y']

#interesting queries...
#"SELECT * FROM rometaxidata WHERE taxi_id =225"
execution_str = "SELECT * FROM rometaxidata WHERE taxi_id =225 AND sim_day_num=1"

#"SELECT * FROM rometaxidata WHERE sim_day_num=1"
#execution_str = "SELECT * FROM rometaxidata WHERE (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)"
#execution_str = "SELECT * FROM rometaxidata WHERE sim_day_num = 10 AND (x BETWEEN -1000 AND 1000) AND (y BETWEEN -1000 AND 1000)" 
#execution_str = "SELECT * FROM rometaxidata WHERE weekday_num = 0 AND taxi_id = 225"
#execution_str = "SELECT DISTINCT taxi_id FROM rometaxidata"

connection = psycopg2.connect(connect_str)

taxidf = pdsql.read_sql_query(execution_str,connection)

#fig, ax = plt.subplots(figsize=(10,15))

fig = plt.gcf()

m = Basemap(
	resolution = 'c',
	projection = 'merc',
	llcrnrlon=12.442, llcrnrlat= 41.856, urcrnrlon=12.5387, urcrnrlat= 41.928)

#(12.442 41.856, 12.5387 41.928, 12.442 41.928, 12.5387 41.856, 12.442 41.856)

#m.fillcontinents(color='#f0f0f0')

road_shp_file = '/home/pdawg/RomeOSMdata/extract/extracted_Roman_roads'

m.readshapefile(road_shp_file, 'roads', drawbounds = True, color='grey')
x, y = m(list(taxidf.long1),list(taxidf.lat1))

plt.scatter(x,y, 5, marker='*', color='red')

#m.plot(taxidf.long1,taxidf.lat1,'b*')

plt.show()

#plt.savefig('/home/pdawg/RomeTaxiData/rome_road_network.png')

