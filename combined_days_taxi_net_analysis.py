# combining multiple day's of rome taxi trace data for network analysis
# pc0179 on the Kboard
# 30.5.2k18


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

#for saving results
import pickle




#useful functions...
def ProcessMapMatchResults(matched_points, timestamps):
# matched_points = result from osrm.match
# output: matched results, and index of points not matched by osrm.match
#longitudes: 0
#latitudes: 1


    nobody_index = []
    matched_ts = []
    matched_pos = []
    for i in range(0,len(matched_points['tracepoints'])):

        if matched_points['tracepoints'][i] is None:
            nobody_index.append(i)
        else:
            matched_ts.append(timestamps[i])
            matched_pos.append(tuple([matched_points['tracepoints'][i]['location'][0],matched_points['tracepoints'][i]['location'][1]]))

    matchedf = pd.DataFrame({'mts':matched_ts,'mpos':matched_pos})
    return matchedf, nobody_index



def haversine_pc(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return Hdistance

def ProcessRouteResults(route_result,start_timestamp,target_timestamp):
# func. to process results from osr.route

    encoded_polyline = route_result['routes'][0]['geometry']
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
# ouput: best estimate of taxi location at time: T
    
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

def Snap2Road(df):
    longitude_2snap = df.longitude.tolist()
    latitude_2snap = df.latitude.tolist()
    snapped_output = df.latitude.tolist()

    for i in range(0,len(latitude_2snap)):
        
        snapped_result = osrm.nearest((longitude_2snap[i],latitude_2snap[i]))
        snapped_output[i] = tuple([snapped_result['waypoints'][0]['location'][0], snapped_result['waypoints'][0]['location'][1]])
    
    df['snap_pos'] = snapped_output
    return df


def MiniLinearDistFilter(linear_dist_mat,min_los_length,max_los_length):
    queck = []
    for row in range(0,len(linear_dist_mat)):
        for col in range(0,row):
            if (linear_dist_mat[row][col]> min_los_length) & (linear_dist_mat[row][col]< max_los_length):
                queck.append([row,col])

    return queck


def HaversineDistPC2(pos1,pos2):
    #where pos1 & pos2 are tuples: (longitude,latitude)
    lon1, lat1, lon2, lat2 = map(np.radians, [pos1[0],pos1[1],pos2[0], pos2[1]])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return int(Hdistance)

def HaversineDistMatPC(input_gps_pos):
# accpets tupple list of positions: [(long1,lat1),(long2,lat2),... etc.]
# outputs numpy array of distances between each of the inputted positions
# only bottom half done due to symmetry
    mat_length = len(input_gps_pos)
    Hdist_matrix = np.zeros((mat_length,mat_length),dtype=int)
    
    for row in range(0,mat_length):
        for col in range(0,row):
            Hdist = HaversineDistPC2(input_gps_pos[row],input_gps_pos[col])
            Hdist_matrix[row,col] = Hdist
    return Hdist_matrix



def CombineTaxiTraceData(days_to_combine):

    # convert to days as in the database numbering system 'trace_day': 0-30
    #days_to_combine = days_to_combine-1

    combined_taxidf = pd.DataFrame()

    for day in days_to_combine:
        start_time_of_day = dt.datetime(2014,2,day,0,0,0)
        start_time_of_day_unix = int(start_time_of_day.timestamp())
        execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE trace_day = %s " % (str(day-1)))
        taxidf1 = pdsql.read_sql_query(execution_str,connection)
        taxidf1 = taxidf1.drop_duplicates()
        taxidf1.taxi_id = taxidf1.taxi_id + (day)*1000
        taxidf1.unix_ts = taxidf1.unix_ts - start_time_of_day_unix
        combined_taxidf = pd.concat([combined_taxidf,taxidf1])

    return combined_taxidf


def TaxiNetworkResults_T(combined_taxidf,T,min_dist,T_search_margin,t_accept):
# split enormous dataframe... to values within T+/- T_search_margin
    #taxidf = pd.DataFrame()
    taxidf = combined_taxidf[(combined_taxidf.unix_ts<T+T_search_margin) & (combined_taxidf.unix_ts>=T-T_search_margin)]

    taxi_ids_before_T = taxidf[taxidf.unix_ts<T].taxi_id.unique()
    taxi_ids_after_T = taxidf[taxidf.unix_ts>=T].taxi_id.unique()

    #taxi_ids2process = set(taxi_ids_after_T).intersection(taxi_ids_before_T)

    taxi_ids_not2process = list(set(taxi_ids_after_T).symmetric_difference(taxi_ids_before_T))
    if len(taxi_ids_not2process)>0:
        for taxi_id_2drop in taxi_ids_not2process:
            taxidf.drop(taxidf[taxidf.taxi_id==taxi_id_2drop].index)

    # this ratio is interesting, early estimates suggest window length of 30s (~91%,
    # maybe better to have 1min , 60s long windows (accept_ratio ~1ish
    # needs more investigation 
    
    print('reject ratio = %f' % float(len(taxi_ids_not2process)/len(taxidf.taxi_id.unique())))
    reject_ratio = (len(taxi_ids_not2process)/len(taxidf.taxi_id.unique()))


    taxis_Tpos = []
    taxis_Tids = []
    taxi_ids_to_process = taxidf.taxi_id.unique().tolist()
    #for each trace, another loop here...
    for taxi_id in taxi_ids_to_process:

        #taxi_pos_estimate = []

        #taxi_id = taxidf.taxi_id[31]

        # map match.
        taxi_subset = taxidf[taxidf.taxi_id==taxi_id].sort_values('unix_ts')

        timestamps2match = taxi_subset.unix_ts.tolist()

        taxi_pos2match = [tuple(x) for x in taxi_subset[['longitude','latitude']].values]

        matched_points = osrm.match(taxi_pos2match, overview="simplified", timestamps=timestamps2match, radius=None)

    # error checking..
        if type(matched_points) is str: # implies no points were matched, hence ditch...
            taxi_pos_estimate = None        
            #snapped_subset = Snap2Road(taxi_subset)
            #matchedf = pd.DataFrame({'mpos':snapped_subset.snap_pos, 'mts':snapped_subset.unix_ts})

        else:
            matchedf, nobody_index = ProcessMapMatchResults(matched_points, timestamps2match)
        
        # quickly remove those where taxi_ts = T

            if (any(matchedf['mts'] == T)) and (bool(np.isnan(matchedf[matchedf.mts==T].mpos.values[0][0])) is False):
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
                # for now, people just do nothing
                if (adf.isnull().any()==True) or (bdf.isnull().any()==True):
                    taxi_pos_estimate = None # [np.nan] #tuple([np.nan,np.nan])

                #if (taxi_pos_estimate is not None) and ((bool(np.isnan(adf.mpos[0])) is True) or (bool(np.isnan(bdf.mpos[1])) is True)):
                    taxi_pos_estimate = None
                #if adf.isnull().any()==True and len(nobody_index)>0:            
                   #taxi_subset[taxi_subset.unix_ts>T]            
        # maybe just snap? then again routing might be a bitch.

                else:

                    d = haversine_pc(adf.mpos[0], adf.mpos[1],bdf.mpos[0],bdf.mpos[1])
                   #if adf.mpos == bdf.mpos:
                    if d<=min_dist:
                        taxi_pos_estimate = adf.mpos

                    else:
                        osrm_route_result = osrm.simple_route([bdf.mpos[0],bdf.mpos[1]],[adf.mpos[0],adf.mpos[1]],output='full',overview="full", geometry='polyline',steps='True',annotations='true')
                        if type(osrm_route_result) is str: taxi_pos_estimate = None
                        else: link_data, route_nodes  = ProcessRouteResults(osrm_route_result,bdf.mts,adf.mts)
                        
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

        if taxi_pos_estimate is not None and (bool(np.isnan(taxi_pos_estimate[0])) is False) and (bool(np.isinf(taxi_pos_estimate)[0]) is False):
            taxis_Tpos.append(taxi_pos_estimate)
            taxis_Tids.append(taxi_id)

    print('succesfull spatial estimation = %f' % (float(len(taxis_Tids)/len(taxi_ids_to_process))))
    reject_taxis_pos = (1 - float(len(taxis_Tids)/len(taxi_ids_to_process)))



    min_los_length = 0
    max_los_length = 100 #same as fucking paper.

    #def TaxisWithinNOLOSRange(input_gps_pos,taxis_Tids,max_los_length,min_los_length):


    input_gps_pos = taxis_Tpos

    # accpets tupple list of positions: [(long1,lat1),(long2,lat2),... etc.]
    # creates a haversine distance matrix, then finds pairs of taxis that are within
    # main_los_length and max_los_length, typical values [0,100]
    # outputs list of taxi_id pairs and their respective haversine distasnces between them
    # [(taxi_id_A,taxi_id_B,t,haversine_distance)]
    mat_length = len(input_gps_pos)
    Hdist_matrix = np.zeros((mat_length,mat_length),dtype=int)
    taxis_nolos = []
    #queck2 = []
    for row in range(0,mat_length):

        for col in range(0,row):
            Hdist = HaversineDistPC2(input_gps_pos[row],input_gps_pos[col])
            Hdist_matrix[row,col] = Hdist

            if (Hdist > min_los_length) & (Hdist < max_los_length):
                taxis_nolos.append((taxis_Tids[row],taxis_Tids[col],input_gps_pos[row],input_gps_pos[col],Hdist))
                #queck2.append([row,col])


    # Line of Sight Model:
    num_buildings = []
    for i in range(len(taxis_nolos)):
        #i=0

        #longitude,latitude in query
        LoS_execution_str = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxis_nolos[i][2][0]),str(taxis_nolos[i][2][1]),str(taxis_nolos[i][3][0]),str(taxis_nolos[i][3][1])))

        LoS_df = pdsql.read_sql_query(LoS_execution_str,connection)

        num_buildings.append(len(LoS_df))



    if len(taxis_nolos)>0: 
        taxiAid, taxiBid, Alonglat, Blonglat, Hdist = zip(*taxis_nolos)
        RESULT_DF = pd.DataFrame({'taxiAid':taxiAid,'taxiBid':taxiBid,'Alonglat':Alonglat,'Blonglat':Blonglat,'Hdist':Hdist,'num_buildings':num_buildings})


    #print('current search time: %i iterations left: %f' % (T, ((T_search_times[-1]-T)/T_search_margin)))


    return RESULT_DF, reject_ratio, reject_taxis_pos


#Connection to database
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)


#Time of interest:

# all mondays...
Mondays_to_combine = [3,10,17,24] # which days in february 2014
Tuesdays_to_combine = [4,11,18,25]
Wednesdays_to_combine = [5,12,19,26]
Thursdays_to_combine = [6,13,20,27]
Fridays_to_combine = [7,14,21,28]

#all_days_to_combine = [[3,10,17,24],[4,11,18,25]] # due to running on small desktop... , Mon,Tues DONE.

# now running on mega desktop: weds, thurs, fridays...
all_days_to_combine = [[5,12,19,26],[6,13,20,27],[7,14,21,28]]


T_search_times = list(range(30,(60*60*24)-30,30))

#random little test...
#T_search_times = list(range(30,180,30))



T_search_margin = 15#30 compare 60s windows with 50% overlap to 30s windows no overlap.
t_accept = 1 #second either side? just use this value for position

min_dist = 5 # 15 #minimum distance in metres actually worth interpolating...

# skeleton of main loop...

import timeit
start_script_time = timeit.default_timer()


for I in range(len(all_days_to_combine)):

    combined_taxidf = CombineTaxiTraceData(all_days_to_combine[I])

    RESULT_DICT = {}
    reject_taxis_pos_list = []
    reject_ratio_list = []
    OVERALL_reject_stats = []

    for T in T_search_times:

        RESULT_DF, reject_ratio, reject_taxis_pos = TaxiNetworkResults_T(combined_taxidf,T,min_dist,T_search_margin,t_accept)

        RESULT_DICT[T] = RESULT_DF
        reject_ratio_list.append(reject_ratio)
        reject_taxis_pos_list.append(reject_taxis_pos)

    OVERALL_reject_stats.append([np.mean(reject_ratio),np.median(reject_ratio),np.mean(reject_taxis_pos),np.median(reject_taxis_pos)])



    with open(('massive_combined_%i_.pickle' % (int(I))),'wb') as handle:
        pickle.dump(RESULT_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(('reject_results_%i_.pickle' % (int(I))), 'wb') as handle:
        pickle.dump(OVERALL_reject_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)




stop_script_time = timeit.default_timer()
total_script_running_time = stop_script_time - start_script_time

mins, secs = divmod(total_script_running_time, 60)
hours, mins = divmod(mins, 60)

print("Total running time: %d:%d:%d.\n" % (hours, mins, secs))




