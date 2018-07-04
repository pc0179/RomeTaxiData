"""
pc0179 on the Kboard... 
3.7.2k18
script for Rome,
idea is to build location and VANET pre-computed dicts/pickles for NetCar simulations...
eventually to compare with SF...
"""

import numpy as np
import pandas as pd
import pickle
import osrm
import polyline
import psycopg2
import pandas.io.sql as pdsql
from sqlalchemy import create_engine

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


def ProcessRouteResults(route_result,start_timestamp,target_timestamp):
# func. to process results from osr.route

    encoded_polyline = route_result['routes'][0]['geometry']
    route_nodes = pd.DataFrame(polyline.decode(encoded_polyline), columns=['latitude','longitude'])
    link_data = pd.DataFrame({'distance':route_result['routes'][0]['legs'][0]['annotation']['distance'], 'duration': route_result['routes'][0]['legs'][0]['annotation']['duration'], 'dur_cumsum' : (((np.cumsum(route_result['routes'][0]['legs'][0]['annotation']['duration'])/np.sum(route_result['routes'][0]['legs'][0]['annotation']['duration']))*(target_timestamp-start_timestamp))+start_timestamp)})
# this last section, allows for temporal scaling, i.e. if the start and end times don't match what
# osrm predicts for the journey, the real time of the taxi is split in proportion to the line segments predict journey time by osrm.
    return link_data, route_nodes


def haversine_pc(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return Hdistance


def Straight_Line_Interp(x1,y1,t1,x2,y2,t2,T):
# strictly moving from x1,y1 --> x2,y2,...
# t1<t2.
    dt = t2-t1
    dT = T-t1
    xT = dT*(x2-x1)/dt + x1
    yT = dT*(y2-y1)/dt + y1

    return round(xT,6),round(yT,6)

def TaxiPositionEstimate_at_T(taxidf,T,min_dist,t_accept):
# split enormous dataframe... to values within T+/- T_search_margin
    #taxidf = pd.DataFrame()

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
    if len(taxidf.taxi_id.unique())>0:
        print('reject ratio = %f' % float(len(taxi_ids_not2process)/len(taxidf.taxi_id.unique())))
        reject_ratio = (len(taxi_ids_not2process)/len(taxidf.taxi_id.unique()))

    else:
        print('reject ratio = ALL rejected')
        reject_ratio = 1


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


                if (adf.isnull().any()==True) or (bdf.isnull().any()==True):
                    taxi_pos_estimate = None # [np.nan] #tuple([np.nan,np.nan])

                #if (taxi_pos_estimate is not None) and ((bool(np.isnan(adf.mpos[0])) is True) or (bool(np.isnan(bdf.mpos[1])) is True)):
                    taxi_pos_estimate = None

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


    if len(taxi_ids_to_process)>1:
        print('succesfull spatial estimation = %f' % (float(len(taxis_Tids)/len(taxi_ids_to_process))))
        reject_taxis_pos = (1 - float(len(taxis_Tids)/len(taxi_ids_to_process)))

    else:
        print('no success at estimating position')
        reject_taxis_pos = 1

    return taxis_Tpos, taxis_Tids, reject_ratio, reject_taxis_pos




def HaversineDistPC2(pos1,pos2):
    #where pos1 & pos2 are tuples: (longitude,latitude)
    lon1, lat1, lon2, lat2 = map(np.radians, [pos1[0],pos1[1],pos2[0], pos2[1]])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return int(Hdistance)

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000

    haversine_distance_metres = np.round(m)

    return haversine_distance_metres


def LineOfSightModel(input_gps_pos, taxis_Tids, min_los_length, max_los_length):

    # accpets tupple list of positions: [(long1,lat1),(long2,lat2),... etc.]
    # creates a haversine distance matrix, then finds pairs of taxis that are within
    # min_los_length and max_los_length, typical values [0,100]
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


    return taxis_nolos, num_buildings


T_search_margin = 30 #seconds either side of T_search, 30 yields too few points, 150 seems to achieve 80%+....
T_sample_frequency = 30
T_accept = 1 # i.e. 1 second either side... no need to interp..., just accept result.

min_dist = 200 #metres, same as porto paper

#Connection to database
#connect_str = "dbname ='sfdata_mike' user='postgres' host='localhost' password='postgres'"
#connection = psycopg2.connect(connect_str)
#connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'"
connection = psycopg2.connect(connect_str)



#2. Quick filter, db data to pandas dataframe
#execution_str = ("SELECT unix_ts FROM rome_taxi_trace")
#entire_rome_taxidf = pdsql.read_sql_query(execution_str,connection)
#entire_rome_taxidf = taxidf.drop_duplicates() #removes duplicates, an ongoing problem.
# sooooo hacky......

trace_start_time_unix = 1392681660 # Tuesday 18th Feb. 2014 
trace_end_time_unix = trace_start_time_unix + (60*60*24) #ie 24 hours...

T_search_times = list(range(trace_start_time_unix,trace_end_time_unix,T_sample_frequency))

reject_ratio_list = []
reject_taxis_pos_list = []

VANET_DICT = {}
POSITION_DICT = {}


import timeit
start_script_time = timeit.default_timer()



shitty_progress_counter = 0
save_to_pickle_num = 0

min_los_length = 0
max_los_length = 200 #same as fucking paper.



for T in T_search_times:

    execution_str = ("SELECT taxi_id,unix_ts,latitude,longitude FROM rome_taxi_trace WHERE unix_ts BETWEEN %s AND %s " % (str(T-T_search_margin),str(T+T_search_margin)))

    #rometaxidf = entire_rome_taxidf[entire_rome_taxidf.unix_ts<T+T_search_margin) & (entire_rome_taxidf.unix_ts>=T-T_search_margin)]

    rometaxidf = pdsql.read_sql_query(execution_str,connection)
    rometaxidf = rometaxidf.drop_duplicates()

#filter here? something to remove all taxis parked at depot in SF (-122.39474, 37.75160)
    #sftaxidf = FilterDFRowsBlobDist(sftaxidf,SF_taxi_depot_location)
    
    taxis_Tpos, taxis_Tids, reject_ratio, reject_taxis_pos = TaxiPositionEstimate_at_T(rometaxidf,T,min_dist,T_accept)

    taxis_nolos, num_buildings = LineOfSightModel(taxis_Tpos, taxis_Tids, min_los_length, max_los_length)


    if len(taxis_nolos)>0: 
        taxiAid, taxiBid, Alonglat, Blonglat, Hdist = zip(*taxis_nolos)
        vanet_df = pd.DataFrame({'taxiAid':taxiAid,'taxiBid':taxiBid,'Alonglat':Alonglat,'Blonglat':Blonglat,'Hdist':Hdist,'num_buildings':num_buildings})

    else: vanet_df = None


    if len(taxis_Tpos)>0: position_df = pd.DataFrame({'taxi_id':taxis_Tids,'longlat':taxis_Tpos})

    else: position_df = None



    VANET_DICT[T] = vanet_df

    POSITION_DICT[T] = position_df


    reject_ratio_list.append(reject_ratio)
    reject_taxis_pos_list.append(reject_taxis_pos)

    shitty_progress_counter +=1
    print('progress: %f' % (shitty_progress_counter/len(T_search_times)))


    save_to_pickle_num = shitty_progress_counter%50
    if save_to_pickle_num<1:

        with open(('/home/user/RomeTaxiData/pre_computed_data/T0_to_T%s_Rome_taxi_positions_dict.pickle' % (str(shitty_progress_counter))),'wb') as handle:
            pickle.dump(POSITION_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('data saved to pickle!')

        with open(('/home/user/RomeTaxiData/pre_computed_data/T0_to_T%s_Rome_vanet_dict.pickle' % (str(shitty_progress_counter))),'wb') as handle:
            pickle.dump(VANET_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('data saved to pickle!')



OVERALL_reject_stats = [np.mean(reject_ratio_list),np.median(reject_ratio_list),np.mean(reject_taxis_pos_list),np.median(reject_taxis_pos_list)]

with open('feb18_Rome_taxis_positions_dict.pickle','wb') as handle:
    pickle.dump(POSITION_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('feb18_Rome_taxis_vanet_dict.pickle','wb') as handle:
    pickle.dump(VANET_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('feb18_Rome_reject_stats_dict.pickle','wb') as handle:
    pickle.dump(OVERALL_reject_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)



stop_script_time = timeit.default_timer()
total_script_running_time = stop_script_time - start_script_time

mins, secs = divmod(total_script_running_time, 60)
hours, mins = divmod(mins, 60)

print("Total running time: %d:%d:%d.\n" % (hours, mins, secs))


