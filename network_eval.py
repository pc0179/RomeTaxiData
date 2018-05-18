# pc0179 on the Kboard
# attempt to analyse an entire day of rome taxi trace data
"""
pick one day, 2014,2,3... 00:00 to 23:59
for every 1s? count how many taxis 'can' talk
- count those just within 100m of each other
- count those within LoS + 100m of each other and compare
"""
import matplotlib
matplotlib.use('Agg')


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

#some useful-ish functions:
def MiniLinearDistFilter(linear_dist_mat,min_los_length,max_los_length):
    queck = []
    for row in range(0,len(linear_dist_mat)):
        for col in range(0,row):
            if (linear_dist_mat[row][col]> min_los_length) & (linear_dist_mat[row][col]< max_los_length):
                queck.append([row,col])

    return queck

def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.
    dt = t2-t1
    dT = T-t1
    xT = dT*(x2-x1)/dt + x1
    yT = dT*(y2-y1)/dt + y1

    return round(xT,6),round(yT,6)

def haversine_dist_matrix(latitudes,longitudes):
    
    x = np.asarray(longitudes)
    m, k = x.shape
    y = np.asarray(latitudes)
    n, kk = y.shape
    result = np.empty((m,n),dtype=float)

    if m<n:
        for i in range(m):
           result[i,:] = spherical_dist(x[i],y)
    else:
        for j in range(n):
            result[:,j] = spherical_dist(x,y[j])

    return result

def spherical_dist(pos1, pos2, r=3958.75e3):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def Route_Osrm(start_long,start_lat,target_long,target_lat,start_timestamp,target_timestamp):

    route_result = osrm.simple_route([start_long,start_lat],[target_long,target_lat],output='full',overview="full", geometry='polyline',steps='True',annotations='true')

    encoded_polyline = route_result['routes'][0]['geometry']

#plot_route_nodes = pd.DataFrame(polyline.decode(encoded_polyline))

    route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['lat','long'])

    link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})
# this last section, allows for temporal scaling, i.e. if the start and end times don't match what
# osrm predicts for the journey, the real time of the taxi is split in proportion to the line segments predict journey time by osrm.

    return link_data, route_nodes

def Straight_Line_Distance(x1,y1,x2,y2):
    d = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return d

#~--- now for some serious scripting.........................................
#1. Query taxi trace database for trace data at a particular time, T

#Time of interest:
T_date = dt.datetime(2014,2,3,12,30,25)
T_unix = int(T_date.timestamp())
t_margin = 30 #in seconds..., i.e. a minute eitherside??? note taxi_id = 8 produces interesting...
t_mini_margin = 2 # seconds either side, where we accept the current result

#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)

# probably start the main loop here... for seconds in day...
#fur mike-pc
execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude,x,y FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T_unix-t_margin),str(T_unix+t_margin)))


start_time = dt.datetime(2014,2,3,6,0,0)
Tstart_unix = int(start_time.timestamp())

T_search_times = list(range(Tstart_unix,Tstart_unix+(60*60*13),30)) #search for 14 hours? at every 30 seconds, this is a lot of queries... moving on...
T_search_margin = 15 #i.e. thirty second chunks
t_accept = 1 #second either side? just use this value for position



num_real_connections = []
num_unreal_connections = []
total_conns = []

for T in T_search_times:

    execution_str = execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude,x,y FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin))) 
    #2. Quick filter, db data to pandas dataframe
    taxidf = pdsql.read_sql_query(execution_str,connection)
    taxidf = taxidf.drop_duplicates() #removes duplicates, an ongoing problem.

    # taxis within t_mini_margin, are assumed to be correct, no further processing here
    prime_taxis = taxidf.loc[(taxidf.unix_ts>T-t_accept) & (taxidf.unix_ts<T+t_accept)]
    #add values to 'final' results dataframe
    estimate_taxis_positions = pd.DataFrame({'latitude':list(prime_taxis.latitude),'longitude':list(prime_taxis.longitude)}, index=list(prime_taxis.taxi_id))

    #remove from processing dataframe those taxis who have a timestamp within t range
    taxidf2 = taxidf.drop(taxidf[(taxidf.unix_ts>T-t_accept) & (taxidf.unix_ts<T+t_accept)].index)

    taxidf2['ts_diff'] = taxidf2.unix_ts-T

    beforedf = taxidf2[taxidf2['unix_ts']<T]
    afterdf = taxidf2[taxidf2['unix_ts']>T]

    before_taxi_ids = list(beforedf.taxi_id.unique())
    after_taxi_ids = list(afterdf.taxi_id.unique())
    taxi_ids = set(before_taxi_ids).intersection(after_taxi_ids)

    #3. loop through all taxi_id's estimating their position at chosen time, T
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
            D_min = 5 # 15 #minimum distance in metres actually worth interpolating...
            if d<D_min:
        #taxi_position = [bdf2.long1,bdf2.lat1] #maybe in future use nearest value...
                #xT,yT = Straight_Line_Interp(adf2.x,adf2.y,adf2.unix_ts,bdf2.x,bdf2.y,bdf2.unix_ts,T)
                apprx_txi_lat1.append(bdf2.latitude)
                apprx_txi_long1.append(bdf2.longitude)
                apprx_txi_id.append(taxi_id)


            else: 
                link_data, route_nodes  = Route_Osrm(bdf2.longitude,bdf2.latitude,adf2.longitude,adf2.latitude,bdf2.unix_ts,adf2.unix_ts)
                #this is a rather ugly fix, to the problem of null routes, given very short distances between points
                if link_data.distance[0] <D_min:
                    apprx_txi_lat1.append(route_nodes.lat[0])
                    apprx_txi_long1.append(route_nodes.long[0])
                    apprx_txi_id.append(taxi_id)
                else: 
                    a = link_data[link_data.dur_cumsum>T].dur_cumsum.idxmin()
	    # herein at a, lies an interesting problem. There are occassions where the vehicle travels exceedingly slowly, therefore it's start and end time  do not match the times predicted by osrm (to get from A-->B), i.e. osrm_generated_timestamp_at_B < actual timestamp at b, therefore need to think about scaling, such that the duration cumsum takes into account the discrepency between actuall_end_timestamp and the one predicted by osrm.
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


                    xT,yT = Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T)

                    apprx_txi_lat1.append(yT)
                    apprx_txi_long1.append(xT)
                    apprx_txi_id.append(taxi_id)



    #4. results from loop, estimate linear distance between taxis for initial sorting...
    # regards Line of Sight (LoS) model
    apprx_txi_pos_df = pd.DataFrame({'latitude':apprx_txi_lat1,'longitude':apprx_txi_long1}, index=apprx_txi_id) #uses taxi id as index... could be interesting

    estimate_taxis_positions = estimate_taxis_positions.append(apprx_txi_pos_df)


    #5. Line of Sight Model

    #linear_dist_df = pd.DataFrame(haversine_dist_matrix(estimate_taxis_positions.values, estimate_taxis_positions.values),index=estimate_taxis_positions.index, columns=estimate_taxis_positions.index)
    linear_dist_mat = np.array(haversine_dist_matrix(estimate_taxis_positions.values, estimate_taxis_positions.values))



    LoS_results = []
    #los_line_string_list = []
    #los_line_list = []
    #nolos_line_string_list = []
    #nolos_line_list = []
    taxis_los = 0
    taxis_nolos = 0

    min_los_length = 0
    max_los_length = 100 #same as fucking paper.
    nearby_taxis = MiniLinearDistFilter(linear_dist_mat,min_los_length,max_los_length)

    for i in range(0,len(nearby_taxis)):

        
        taxi_a = estimate_taxis_positions.iloc[nearby_taxis[i][0]].tolist()
        taxi_b = estimate_taxis_positions.iloc[nearby_taxis[i][1]].tolist()
        
        LoS_execution_str = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a[1]),str(taxi_a[0]),str(taxi_b[1]),str(taxi_b[0])))

        LoS_df = pdsql.read_sql_query(LoS_execution_str,connection)
        
        LoS_results.append(len(LoS_df))

        # shitty counter, need to know how many pairs are within los/otherwise
        if LoS_results[i]<1:
        
            taxis_los+=1
        else:
            taxis_nolos+=1


    num_real_connections.append(taxis_los)
    num_unreal_connections.append(taxis_nolos)
    total_conns.append(len(nearby_taxis))
    print('current time batch = %i' % (T))
    print('progress-ish = %f' % (len(num_real_connections)/len(T_search_times)))



end_script_time = time.time()

#create pandas dataframe for easy saving to fucking csv.


time_4_plot = np.array(T_search_times[0:len(total_conns)])-Tstart_unix

RESULTS = pd.DataFrame({'time_ts': time_4_plot,'los':num_real_connections,'nolos':num_unreal_connections,'total_conns':total_conns})

RESULTS.columns = ['time_ts','los','nolos','total_conns']

RESULTS.to_csv('taxi_net_eval.csv',sep=',',header=True,index=False)


#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.figure()
plt.plot(time_4_plot,num_real_connections,'-*k',time_4_plot,num_unreal_connections,'--ob')
#plt.plot(T_search_times[0:len(total_conns)]-Tstart_unix,num_real_connections,'*k',T_search_times[0:len(total_conns)]-Tstart_unix,num_unreal_connections,'ob')
plt.xlabel('Time/s')
plt.ylabel('Number of poten. conns. between taxis')
plt.savefig('taxi_conns_1day.pdf', dpi=400)

#"""
#0. define time slots/sample windows for processsssing
#1. query db for taxis within defined time slot
#2. FILTER
#    z) remove duplicates...
#    a) taxis within a second of T ==> just use their current position
#    b) taxis whose position does not change between T-t and T+t, use that posision... 
#    c) map-match to road segment? or does route accomplish this already?...
#    d) maybe should check if points are within a buliding to start?... 


#once nearest taxis are established, then run LoS model
#- how many taxis can actually (LoS) communicate at that particular second?
#- how many can't communicate (NoLoS) but, are within 100m range at time, T?
#- number of closest commuicable neighbours?

#note st_intersects does not seem to return a True/False statement... instead returns number of buildings (i think)... would it be faster with T/F instead?
#note iloc returns literal position within pd.df, whereas loc uses the index provided (i.e. the taxi_ids in some cases...)

#just ran.
#very slow. around 5-10mins for 90mins-ish trace data analysis...
#once again, problems when link_data = 0 (i.e has no values) line 182, d =13 metres, maybe need to up it....
#"""
