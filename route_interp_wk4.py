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
import matplotlib.pyplot as plt

#---------------------------------------
# 0. some personal functions
def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.

	dt = t2-t1
	dT = T-t1
	xT = dT*(x2-x1)/dt + x1
	yT = dT*(y2-y1)/dt + y1

	return round(xT,6),round(yT,6)


def Route_Osrm(start_long,start_lat,target_long,target_lat,start_timestamp,target_timestamp):

	route_result = osrm.simple_route([start_long,start_lat],[target_long,target_lat],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

	encoded_polyline = route_result['routes'][0]['geometry']

#plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

	route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['lat','long'])

	link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})

	return link_data, route_nodes

def Straight_Line_Distance(x1,y1,x2,y2):
	d = ((x1-x2)**2 +(y1-y2)**2)**0.5
	return d

#---------------------------------------------------------------------
#1. Querying Database

#User Picks Time T
T_date = dt.datetime(2014,2,3,12,30,25)
T_unix = int(T_date.timestamp())
t_margin = 30 #in seconds..., i.e. a minute eitherside??? note taxi_id = 8 produces interesting...

connect_str = "dbname='rometaxitraces' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)

#execution_str = ("SELECT taxi_id,unix_ts,lat1,long1 FROM rometaxidata WHERE sim_day_num =%s" % (str(sim_day_num)))

#execution_str = ("SELECT taxi_id,unix_ts,lat1,long1, abs(unix_ts - %s) as d FROM rometaxidata order by d limit 10;" % (str(T_unix)))

execution_str = ("SELECT taxi_id,unix_ts,lat1,long1,x,y FROM rometaxidata WHERE unix_ts BETWEEN %s AND %s " % (str(T_unix-t_margin),str(T_unix+t_margin)))

taxidf = pdsql.read_sql_query(execution_str,connection)

taxidf = taxidf.drop_duplicates()

t_mini_margin = 2 # seconds either side, where we accept the current result
# really need to get better with using indexing, rather than continuousy re-assigning dataframes
prime_taxis = taxidf.loc[(taxidf.unix_ts>T_unix-t_mini_margin) & (taxidf.unix_ts<T_unix+t_mini_margin)]


estimate_taxis_positions = pd.DataFrame({'lat1':list(prime_taxis.lat1),'long1':list(prime_taxis.long1)}, index=list(prime_taxis.taxi_id))


#taxidf2 = taxidf.drop(taxidf[taxidf.unix_ts==T_unix].index)
taxidf2 = taxidf.drop(taxidf[(taxidf.unix_ts>T_unix-t_mini_margin) & (taxidf.unix_ts<T_unix+t_mini_margin)].index)

# add diff column... (unix_ts - user set time, T)
taxidf2['ts_diff'] = taxidf2.unix_ts-T_unix

beforedf = taxidf2[taxidf2['unix_ts']<T_unix]
afterdf = taxidf2[taxidf2['unix_ts']>T_unix]

#taxi_ids = taxidf2.taxi_id.unique()

before_taxi_ids = list(beforedf.taxi_id.unique())
after_taxi_ids = list(afterdf.taxi_id.unique())
taxi_ids = set(before_taxi_ids).intersection(after_taxi_ids)


apprx_txi_lat1 = []
apprx_txi_long1 = []
apprx_txi_id = []

for taxi_id in taxi_ids:

	bdf = beforedf[beforedf['taxi_id']==taxi_id] #taxi_ids[8]]
	adf = afterdf[afterdf['taxi_id']==taxi_id] #taxi_ids[8]]
	if (len(adf)+len(bdf))>2:
		
		bdf2 = bdf.loc[bdf['ts_diff'].idxmax()]
		adf2 = adf.loc[adf['ts_diff'].idxmin()]

		# maybe somewhere about here, need to make some checks
		# if distance between chosen points is <D_min: pick whatever is closest
		# else: do some routing and interping between points...
		d = Straight_Line_Distance(adf2.x,adf2.y,bdf2.x,bdf2.y)

		D_min = 10 # minimum distance in metres actually worth interpolating...
		if d<D_min:
	#		taxi_position = [bdf2.long1,bdf2.lat1] #maybe in future use nearest value...
			apprx_txi_lat1.append(bdf2.lat1)
			apprx_txi_long1.append(bdf2.long1)
			apprx_txi_id.append(taxi_id)


		else: 
			link_data, route_nodes  = Route_Osrm(bdf2.long1,bdf2.lat1,adf2.long1,adf2.lat1,bdf2.unix_ts,adf2.unix_ts)
			#link_data.dur_cumsum = link_data.duration.cumsum() + bdf2.unix_ts
	
		
			#interp_points_index = link_data.loc[:,'dur_cumsum']>T_unix	

			a = link_data[link_data.dur_cumsum>T_unix].dur_cumsum.idxmin() #interp_points_index[interp_points_index ].index[0]

	# herein at a, lies an interesting problem. There are occassions where the vehicle travesl exceedingly slowly, therefore it's start and end time  do not match the times predicted by osrm (to get from A-->B), i.e. osrm_generated_timestamp_at_B < actual timestamp at b, therefore need to think about scaling, such that the duration cumsum takes into account the discrepency between actuall_end_timestamp and the one predicted by osrm.


			b = a+1

			x1 = route_nodes['long'][a]
			y1 = route_nodes['lat'][a]
			if a==0:
				t1 = link_data['dur_cumsum'][a]-link_data.duration[a]
			else:
				t1 = link_data['dur_cumsum'][a-1]

			x2 = route_nodes['long'][b]
			y2 = route_nodes['lat'][b]
			t2 = link_data['dur_cumsum'][a]


			xT,yT = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T_unix)

			apprx_txi_lat1.append(yT)
			apprx_txi_long1.append(xT)
			apprx_txi_id.append(taxi_id)

"""
plt.plot(route_nodes.long,route_nodes.lat,'+-k')
plt.plot(x1,y1,'dg')
plt.plot(x2,y2,'dr')
plt.plot(xT,yT,'ob') #show be inbetween other two points.
plt.show()
"""


#apprx_txi_pos_df = pd.DataFrame({'taxi_id':apprx_txi_id, 'lat1':apprx_txi_lat1,'long1':apprx_txi_long1})
apprx_txi_pos_df = pd.DataFrame({'lat1':apprx_txi_lat1,'long1':apprx_txi_long1}, index=apprx_txi_id) #uses taxi id as index... could be interesting

estimate_taxis_positions = estimate_taxis_positions.append(apprx_txi_pos_df)


#---------
# Line-of-Sight attempt - LoS
# calculate straight line distance between points at time T.

#radical use of scipy library, does not take into account longs/lats, maybe best to convert to x,y then compute distances... 
from scipy.spatial import distance_matrix
dist_mat = pd.DataFrame(distance_matrix(estimate_taxis_positions.values, estimate_taxis_positions.values),index=estimate_taxis_positions.index, columns=estimate_taxis_positions.index)

#dist_mat.index[dist_mat[::]<0.01].tolist()

#route distance table... from osrm...

table_input_df_cols = ['long1','lat1'] #change the order, long,lat for table query input
estimate_taxis_positions = estimate_taxis_positions[table_input_df_cols]
table_input_coords = estimate_taxis_positions[0:4].values.tolist()

#this does not seem to work... probably need to remove & infront of annotations in python-osrm.core.py, still something fishy here lies with OSRM backend
dist_matrix, snapped_coords = osrm.table(table_input_coords,output='dataframe', send_as_polyline=False, annotations='distance')



'''
# in theory, it starts with a query to db: select tax_id, lat1, long1, t, NEAREST Group by taxi_id?
# if near enough points.. go for it, otherwise need to group by taxi_id, and process each 
# individual trace, ie, the two nearest points to time T
# ie query result = taxi_id_I, x1,y1,t1 and x2,y2,t2... for all taxi_ids within range of query

# processing, requires (map-matching?---ssshh!) then route between points, interpolate, find best estimate
T_query = 1525082338+400 #eg some unix time or date-time obj.
t1 = 1525082338 # earliest poitn, t1<t2... dummy number, actuall number will be returned from initial point (x1,y1,t1)

T_query = 21
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

xT, yT = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T_query)

#xT,yT = Straight_Line_Interp(route_nodes[a][0],route_nodes[a][1],link_data['dur_cumsum'][a],route_nodes[b][0],route_nodes[b][1],link_data['dur_cumsum'][b], T_query)







plt.plot(plot_route_nodes[0],plot_route_nodes[1],'+-k')
plt.plot(x1,y1,'dg')
plt.plot(x2,y2,'dr')
plt.plot(xT,yT,'ob') #show be inbetween other two points.
plt.show()

'''

"""
for taxi_id in taxi_ids:
	
	searchdf = afterdf.loc[afterdf['taxi_id']==taxi_ids[0] & afterdf['ts_diff'].max()]
	searchdf = afterdf[afterdf['taxi_id'] == taxi_ids[0]]
	df3 = taxidf2.iloc[searchdf.ts_diff.idxmin()]
	 
	#nearst t1, and positional values before T_unix,:
	#taxidf2.taxi_id.iloc[(taxidf2['unix_ts']-T_unix).abs().argsort()[:2]]

# find x2,y2,t2, i.e. values nearest but after T_unix:	

df3.iloc[(afterdf.ts_diff.min())]
#taxidf3 = taxidf2[~taxidf2.duplicated(subset=['taxi_id'],keep=False)]
"""

# potential strategy:
# query for ALL taxis within time frame --> pandas dataframe
# within dataframe, elimate duplicates...
# wihtin dataframe elimate those with exact matching times (maybe within 2-3second range), i.e. T = unix_t for taxi_ID...
# for remaining taxis... pair them up (i.e. remove unique 'taxi_ID@ values...)
# for paired taxis, about time T, then apply following interp...
# maybe just get list of unique taxi_ids...
# for each id, find a time before and after T_unix -> quickly route.. and interp...


