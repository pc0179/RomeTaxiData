#pc0179, python 3.5+. yeah deal with it.

#general:
import numpy as np
import pandas as pd
import datetime as dt

#For routing and querying database:
import osrm
import polyline
import psycopg2
import pandas.io.sql as pdsql
from sqlalchemy import create_engine

#For Plotting:
#import matplotlib.pyplot as plt

#---------------------------------------

#1. Querying Database

#User Picks Time T
T_date = dt.datetime(2014,2,3,12,30,25)
T_unix = int(T_date.timestamp())
t_margin = 30 #in seconds..., i.e. a minute eitherside???

connect_str = "dbname='rometaxitraces' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)

#execution_str = ("SELECT taxi_id,unix_ts,lat1,long1 FROM rometaxidata WHERE sim_day_num =%s" % (str(sim_day_num)))

#execution_str = ("SELECT taxi_id,unix_ts,lat1,long1, abs(unix_ts - %s) as d FROM rometaxidata order by d limit 10;" % (str(T_unix)))

execution_str = ("SELECT taxi_id,unix_ts,lat1,long1 FROM rometaxidata WHERE unix_ts BETWEEN %s AND %s " % (str(T_unix-t_margin),str(T_unix+t_margin)))

taxidf = pdsql.read_sql_query(execution_str,connection)

taxidf = taxidf.drop_duplicates()

precise_location_taxis = taxidf[taxidf['unix_ts']==T_unix]
# okay but doesn't remove taxis already removed because we have the precise location..


taxidf2 = taxidf.drop(taxidf[taxidf.unix_ts==T_unix].index)



# potential strategy:
# query for ALL taxis within time frame --> pandas dataframe
# within dataframe, elimate duplicates...
# wihtin dataframe elimate those with exact matching times (maybe within 2-3second range), i.e. T = unix_t for taxi_ID...
# for remaining taxis... pair them up (i.e. remove unique 'taxi_ID@ values...)
# for paired taxis, about time T, then apply following interp...


def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.

	dt = t2-t1
	dT = T-t1
	xT = dT*(x2-x1)/dt + x1
	yT = dT*(y2-y1)/dt + y1

	return round(xT,6),round(yT,6)


def Route_Interp(A,B):
	
route_result = osrm.simple_route([12.442,41.856],[12.5387,41.928],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

encoded_polyline = route_result['routes'][0]['geometry']

plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

route_nodes = polyline.decode(encoded_polyline)

link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])})

# in theory, it starts with a query to db: select tax_id, lat1, long1, t, NEAREST Group by taxi_id?
# if near enough points.. go for it, otherwise need to group by taxi_id, and process each 
# individual trace, ie, the two nearest points to time T
# ie query result = taxi_id_I, x1,y1,t1 and x2,y2,t2... for all taxi_ids within range of query

# processing, requires (map-matching?---ssshh!) then route between points, interpolate, find best estimate
T_query = 1525082338+400 #eg some unix time or date-time obj.
t1 = 1525082338 # earliest poitn, t1<t2... dummy number, actuall number will be returned from initial point (x1,y1,t1)

T_query = 1020.2
t1 = 0
link_data['dur_cumsum'] = link_data['dur_cumsum'] + t1

interp_points_index = link_data.loc[:,'dur_cumsum']>T_query

# actually a bi useless when you think about it: a = interp_points_index[~interp_points_index ].index[-1]
a = interp_points_index[interp_points_index ].index[0]
b = a+1

x1 = route_nodes[a][0]
y1 = route_nodes[a][1]
t1 = link_data['dur_cumsum'][a-1]

x2 = route_nodes[b][0]
y2 = route_nodes[b][1]
t2 = link_data['dur_cumsum'][a]

xT,yT = mid_point(x1,y1,t1,x2,y2,t2,T_query)
#xT,yT = mid_point(route_nodes[a][0],route_nodes[a][1],link_data['dur_cumsum'][a],route_nodes[b][0],route_nodes[b][1],link_data['dur_cumsum'][b], T_query)

"""
plt.plot(plot_route_nodes[0],plot_route_nodes[1],'+-k')
plt.plot(x1,y1,'dg')
plt.plot(x2,y2,'dr')
plt.plot(xT,yT,'ob') #show be inbetween other two points.
plt.show()
"""

plt.plot(plot_route_nodes[0],plot_route_nodes[1],'+-k')
plt.plot(x1,y1,'dg')
plt.plot(x2,y2,'dr')
plt.plot(xT,yT,'ob') #show be inbetween other two points.
plt.show()
