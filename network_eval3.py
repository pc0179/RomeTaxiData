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
            matched_pos.append(tuple([matched_points['tracepoints'][i]['location'][0],matched_points['tracepoints'][i]['location'][1]]))

    matchedf = pd.DataFrame({'mts':matched_ts,'mpos':matched_pos})
    return matchedf, nobody_index

#longitudes: 0
#latitudes: 1

def haversine_pc(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return Hdistance

def ProcessRouteResults(route_result,start_timestamp,target_timestamp):

    encoded_polyline = route_result['routes'][0]['geometry']

#plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

    route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['latitude','longitude'])

    link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})
# this last section, allows for temporal scaling, i.e. if the start and end times don't match what
# osrm predicts for the journey, the real time of the taxi is split in proportion to the line segments predict journey time by osrm.

    return link_data, route_nodes

def Straight_Line_Distance(x1,y1,x2,y2):
    d = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return d

def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.
    dt = t2-t1
    dT = T-t1
    xT = dT*(x2-x1)/dt + x1
    yT = dT*(y2-y1)/dt + y1

    return round(xT,6),round(yT,6)



def RouteAndInterp(matchedf,T,min_dist):
# inputs: matched_data_frame for one Taxi_ID, time T, minimum accepted distance
    
    matchedf['ts_dff'] = matchedf['mts']-T
    adf = matchedf[matchedf['ts_dff']>0].min()
    bdf = matchedf[matchedf['ts_dff']<0].max()

    d = haversine_pc(adf.mpos[0], adf.mpos[1],bdf.mpos[0],bdf.mpos[1])
   #if adf.mpos == bdf.mpos:
    if d<=min_dist:
        taxi_pos_estimate = adf.mpos

    else:
        osrm_route_result = osrm.simple_route([bdf.mpos[0],bdf.mpos[1]],[adf.mpos[0],adf.mpos[1]],output='full',overview="full", geometry='polyline',steps='True',annotations='true')
        link_data, route_nodes  = ProcessRouteResults(osrm_route_result,bdf.mts,adf.mts)
        
#maybe another if statement, if link_data.dur_cumsum == T: ..., else:
        T_index = max(link_data[link_data['dur_cumsum']<=T].index.tolist())

        x1 = route_nodes['longitude'][T_index]
        y1 = route_nodes['latitude'][T_index]

        if T_index == 0:
            t1 = link_data['dur_cumsum'][0]-link_data.duration[T_index]
        else:
            t1 = link_data['dur_cumsum'][T_index-1]


        x2 = route_nodes['longitude'][T_index+1]
        y2 = route_nodes['latitude'][T_index+1]
        t2 = link_data['dur_cumsum'][T_index]

        T_longitude,T_latitude = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T)
        
        taxi_pos_estimate = tuple([T_longitude,T_latitude])

    return taxi_pos_estimate


#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)


#Time of interest:
#tuseday 4th feb 2014, 7am-7pm?
start_time = dt.datetime(2014,2,4,7,0,15)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*12),30)) #search for 12 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 30 #i.e. 30 second chunks
t_accept = 1 #second either side? just use this value for position

num_real_connections = []
num_unreal_connections = []
total_conns = []
min_dist = 5 # 15 #minimum distance in metres actually worth interpolating...


# skeleton of main loop...

T = T_search_times[647]

execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 

taxidf = pdsql.read_sql_query(execution_str,connection)

#filter duplicates, you never know
taxidf = taxidf.drop_duplicates()

taxi_TDF = pd.DataFrame() 
#only sort taxis with points either side of T.
#taxidf['ts_diff'] = taxidf['unix_ts']-T

#before_taxi_ids = list(beforedf.taxi_id.unique())
#after_taxi_ids = list(afterdf.taxi_id.unique())
#taxi_ids_2process = set(before_taxi_ids).intersection(after_taxi_ids)
taxi_ids_before_T = taxidf[taxidf.unix_ts<T].taxi_id.unique()
taxi_ids_after_T = taxidf[taxidf.unix_ts>=T].taxi_id.unique()

#taxi_ids2process = set(taxi_ids_after_T).intersection(taxi_ids_before_T)

taxi_ids_not2process = set(taxi_ids_after_T).symmetric_difference(taxi_ids_before_T)
if len(taxi_ids_not2process)>0:
    taxidf.drop(taxidf.taxi_id==taxi_ids_not2process)

# this ratio is interesting, early estimates suggest window length of 30s (~91%,
# maybe better to have 1min , 60s long windows (accept_ratio ~1ish
# needs more investigation 
accept_ratio = 1 - len(taxi_ids_not2process)/len(taxidf.taxi_id.unique())
print('accept_ratio = %f' % float(accept_ratio))

taxi_pos_at_T = []

#for each trace, another loop here...
for taxi_id in taxidf.taxi_id:

    #taxi_id = taxidf.taxi_id[31]

    # map match.
    taxi_subset = taxidf[taxidf.taxi_id==taxi_id].sort_values('unix_ts')

    timestamps2match = taxi_subset.unix_ts.tolist()

    taxi_pos2match = [tuple(x) for x in taxi_subset[['longitude','latitude']].values]

    matched_points = osrm.match(taxi_pos2match, overview="simplified", timestamps=timestamps2match, radius=None)

    matchedf, nobody_index = ProcessMapMatchResults(matched_points, timestamps2match)
    
    # quickly remove those where taxi_ts = T

    if any(matchedf['mts'] == T):
        #taxi_TDF.append([matchedf[matchedf.mts==T]])
        taxi_pos_estimate = matchedf[matchedf.mts==T].mpos.tolist()[0]

    else:
    # route&/interp
        #taxi_pos_estimate = RouteAndInterp(matchedf,T,min_dist)
        matchedf['ts_dff'] = matchedf['mts']-T
        adf = matchedf[matchedf['ts_dff']>0].min()
        bdf = matchedf[matchedf['ts_dff']<0].max()

        #if map-matching doesn't work...
        # complete fail? --> snap coords instead, then route?
        if (adf.isnull().any()==True) or (bdf.isnull().any()==True):
            taxi_pos_estimate = np.nan

# maybe just snap? then again routing might be a bitch.

        else:

            d = haversine_pc(adf.mpos[0], adf.mpos[1],bdf.mpos[0],bdf.mpos[1])
           #if adf.mpos == bdf.mpos:
            if d<=min_dist:
                taxi_pos_estimate = adf.mpos

            else:
                osrm_route_result = osrm.simple_route([bdf.mpos[0],bdf.mpos[1]],[adf.mpos[0],adf.mpos[1]],output='full',overview="full", geometry='polyline',steps='True',annotations='true')
                link_data, route_nodes  = ProcessRouteResults(osrm_route_result,bdf.mts,adf.mts)
                
        #maybe another if statement, if link_data.dur_cumsum == T: ..., else:
                if any(link_data.dur_cumsum<T):
                    T_index = max(link_data[link_data['dur_cumsum']<=T].index.tolist())
                else:
                    T_index = 0            

                x1 = route_nodes['longitude'][T_index]
                y1 = route_nodes['latitude'][T_index]

                if T_index == 0:
                    t1 = link_data['dur_cumsum'][0]-link_data.duration[T_index]
                else:
                    t1 = link_data['dur_cumsum'][T_index-1]


                x2 = route_nodes['longitude'][T_index+1]
                y2 = route_nodes['latitude'][T_index+1]
                t2 = link_data['dur_cumsum'][T_index]

                T_longitude,T_latitude = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T)
                
                taxi_pos_estimate = tuple([T_longitude,T_latitude])

    taxi_pos_at_T.append(taxi_pos_estimate)

taxidf['pos_at_T']=taxi_pos_at_T
#print(est_taxi_pos)




"""
taxi_pos2match = [tuple(x) for x in taxi_subset[['latitude','longitude']].values]
test_line = polyline.encode(taxi_pos2match)

when map-matching returns nulls...
things get interesting.
if both adf & bdf null?
snap what ever coords available?
then try to route anyway?

#longitudes: 0
#latitudes: 1

import pandas as pd
import numpy as np
A = (3,2)
B = np.nan

df = pd.DataFrame({'a':A,'b':B})
if df.isnull().any() is True:
    pandas_is_good = True


        if adf.isnull().any() == True:
            partial_mmatch_index = taxi_subset[taxi_subset['unix_ts']>T].min().index
            #adf = pd.DataFrame({'mpos'=tuple([taxi_subset.longitude[partial_mmatch_index],taxi_subset.latitude[partial_mmatch_index]]),'mts'=taxi_subset.unix_ts[partial_mmatch_index]})
            adf['ts_dff'] = adf.mts - T
        
        if bdf.isnull().any() == True:
            partial_mmatch_index = taxi_subset[taxi_subset['unix_ts']<T].max().index
            #bdf = pd.DataFrame({'mpos'=tuple([taxi_subset.longitude[partial_mmatch_index],taxi_subset.latitude[partial_mmatch_index]]),'mts'=taxi_subset.unix_ts[partial_mmatch_index]})
            bdf['ts_dff'] = bdf.mts - T
"""




