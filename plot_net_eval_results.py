# script to load processed taxi trace data, i.e. after network_eval3.py had a bash at it...


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()


import osrm
import networkx as nx


# get pickle data...
#pickle_file = '30S_windows_no_overlap_result_ver1.pickle'

#pickle_file = '4_till_11-2-2k14_30s_win_no_overlap_BIG_result.pickle'

#pickle_file = 'massive_combined_0_.pickle'

#pickle_file = '21-2-2k14_0700_1900_taxi_result.pickle'

pickle_file = '/home/user/insyncGdrive/early_taxi_network_results/SF_taxi_win150s_every_75s.pickle'

trace_network_data = pd.read_pickle(pickle_file)

timestamps = list(trace_network_data.keys())


# crazy two hop test?


#G = nx.Graph()
#G.add_edges_from([('v1','v2'),('v2','v4'),('v1','v3')])
#G.add_edges_from([('v1','v2'),('v2','v3'),('v1','v3')])

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)

    return [node for node, length in path_lengths.items() if length == n]

#print(neighborhood(G, 'v1', 2))



def NoLoSHopCount(df,n):
    #Input: df.cols = 'Alonglat', 'Blonglat', 'Hdist', 'num_buildings', 'taxiAid', 'taxiBid'
    #Output: list in order of taxiID's of the number of other taxis avaible within 2 hops
    edge_pair_list = list(zip(df.taxiAid[df.num_buildings>0],df.taxiBid[df.num_buildings>0]))

    G = nx.Graph()
    G.add_edges_from(edge_pair_list)

    taxi_nodes_list = list(G.nodes)
    two_hop_count_list = []
    #col_name = ('%ihop' % (n))
    for taxi_node in taxi_nodes_list:

        path_lengths = nx.single_source_dijkstra_path_length(G,taxi_node)
        two_hop_count_list.append(len([taxi_node for taxi_node, length in path_lengths.items() if length == n]))

    return two_hop_count_list


def LoSHopCount(df,n):

    edge_pair_list = list(zip(df.taxiAid[df.num_buildings<1],df.taxiBid[df.num_buildings<1]))

    G = nx.Graph()
    G.add_edges_from(edge_pair_list)

    taxi_nodes_list = list(G.nodes)
    two_hop_count_list = []
    #col_name = ('%ihop' % (n))
    for taxi_node in taxi_nodes_list:

        path_lengths = nx.single_source_dijkstra_path_length(G,taxi_node)
        two_hop_count_list.append(len([taxi_node for taxi_node, length in path_lengths.items() if length == n]))

    return two_hop_count

# nearly there with hopcounts
# maybe parameterise NoLoS/LoS 
#----

# total number of connections/ message exchange oppurtunities

#ax, ay = zip(*test_data.Alonglat.tolist())
#bx, by = zip(*test_data.Blonglat.tolist())

def PlotVANET(df):

    ax, ay = zip(*df.Alonglat.tolist())
    bx, by = zip(*df.Blonglat.tolist())
    num_buildings = test_data.num_buildings.tolist()

    for i in range(len(by)):

        if num_buildings[i]>0:
            plt.plot([ax[i], bx[i]],[ay[i],by[i]],'ro-')

        else:
            plt.plot([ax[i], bx[i]],[ay[i],by[i]],'gd-')


    return plt.show()


#PlotVANET(test_data)


def CentreOfMassOfTaxis(df,buildings):
    """func. to find Centre of Mass of taxis. For LoS only, set buildings=None, for NoLos set buildings to however many you want included..."""

    if buildings is None:
        df = df[df.num_buildings<1]
    else:
        df = df[df.num_buildings>buildings]

    ax, ay = zip(*df.Alonglat.tolist())
    bx, by = zip(*df.Blonglat.tolist())

    Xcofm = (np.sum(ax) + np.sum(bx))/(len(ax)+len(bx))
    Ycofm = (np.sum(ay) + np.sum(by))/(len(ay)+len(by))

    return Xcofm, Ycofm


#test_data = trace_network_data[timestamps[999]]
#CofM_longitude, CofM_latitude = CentreOfMassOfTaxis(test_data,buildings=None)



#http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#from sklearn.cluster import DBSCAN

#X = test_data.
#db = DBSCAN(eps=0.3, min_samples=10).fit(X)



# back to fucking routing/table shit.
def RouteDistanceColumn(df):
    #df = test_data
    #i = 0
    Rdist_list = []
    for i in range(len(df)):

        table_result = osrm.table([df.Alonglat[i],df.Blonglat[i]],[df.Alonglat[i],df.Blonglat[i]],ids_origin=None, ids_dest=None, output='np', minutes=False, send_as_polyline=True, annotations='distance')

        #if table_result == 'Bad Request': rdist = None
        if type(table_result) is str: rdist = None
        else: rdist = int(round(table_result[table_result>0].min()))

        Rdist_list.append(rdist)

    df['Rdist']=Rdist_list
    return df 


#RouteDistanceColumn(test_data)



def InfectionSpreadNoLoS(df,infected_list):
#simple, but in effective, since order of pairs of taxiID's (in df) will change outcome of infected_list
# to avoid this, only one way transmission per iteration, hence new/infection list.

    new_infections = infected_list
    for i in df.index:


        if (df.taxiAid[i] in infected_list) and (df.taxiBid[i] not in infected_list):

            new_infections.append(df.taxiBid[i])

        if (df.taxiBid[i] in infected_list) and (df.taxiAid[i] not in infected_list):

            new_infections.append(df.taxiAid[i])


    return new_infections


def InfectionSpreadLoS(df,infected_list):
#simple, but in effective, since order of pairs of taxiID's (in df) will change outcome of infected_list
# to avoid this, only one way transmission per iteration, hence new/infection list.

    new_infections = infected_list
    for i in df.index:

        if df.num_buildings[i]<1:

            if (df.taxiAid[i] in infected_list) and (df.taxiBid[i] not in infected_list):

                new_infections.append(df.taxiBid[i])

            if (df.taxiBid[i] in infected_list) and (df.taxiAid[i] not in infected_list):

                new_infections.append(df.taxiAid[i])


    return new_infections



test_data = trace_network_data[1211159404]


sorted_timestamps = sorted(trace_network_data.keys(), reverse=False)

plotting_time = []
noLoS_result = []
LoS_result = []

LoS_connections = []
total_connections = []

noLoS_infected_list = ['ayshowg']
LoS_infected_list = ['ayshowg']

#subset of time we're interested in:
# tuseday 20th may - wednesday 20th may 10.00-10ish ~24hours
start_time_index = 3456
end_time_index = start_time_index + 1152


subset_sorted_timestamps = sorted_timestamps[start_time_index:end_time_index]

for timestamp in subset_sorted_timestamps:

    df = trace_network_data[timestamp]

    noLoS_infected_list = InfectionSpreadNoLoS(df, noLoS_infected_list)
    LoS_infected_list = InfectionSpreadLoS(df, LoS_infected_list)

    noLoS_result.append(len(noLoS_infected_list))    
    LoS_result.append(len(LoS_infected_list))
        
    plotting_time.append(timestamp)

    total_connections.append(len(df))
    LoS_connections.append(len(df[df.num_buildings<1]))




#plt.plot(plotting_time,noLoS_result,'-ok', plotting_time, LoS_result, '-db',plotting_time,total_connections,'-k',plotting_time,LoS_connections, '-b')
#plt.ylabel('Number of Infected Taxis')
#plt.xlabel('Sim. Time/s')
#plt.show()



#plt.plot(plotting_time,noLoS_result,'-ok', plotting_time, LoS_result, '-db')
#plt.ylabel('Number of Infected Taxis')
#plt.xlabel('Sim. Time/s')
#plt.show()


###### normalised infection spreading plots.


norm_time = (np.array(plotting_time[start_time_index:end_time_index])-plotting_time[start_time_index])/plotting_time[end_time_index]

norm_time = (np.array(plotting_time)-plotting_time[0])/(plotting_time[-1]-plotting_time[0])

norm_noLoS_result = np.array(noLoS_result)/max(noLoS_result)
norm_LoS_result = np.array(LoS_result)/max(LoS_result)


plt.plot(norm_time,norm_noLoS_result,'-ok',norm_time,norm_LoS_result,'-db')
plt.ylabel('Ratio of Infected Taxis')
plt.xlabel('Normalised Time')
plt.show()





####### general location/clustering of where the hell these taxis communicate with eachother



LoS_points_list = []
NOLoS_points_list = []


for timestamp in subset_sorted_timestamps:

    df = trace_network_data[timestamp]

    for i in df.index:

        if df.num_buildings[i]>0: NOLoS_points_list.append([[df.Alonglat[0][0],df.Alonglat[0][1]],[df.Blonglat[0][0],df.Blonglat[0][1]]])

        if df.num_buildings[i]<1: LoS_points_list.append([[df.Alonglat[0][0],df.Alonglat[0][1]],[df.Blonglat[0][0],df.Blonglat[0][1]]])

        

import shapefile

#W = shapefile.Writer(shapefile.POLYLINE)
#W.line(parts= [[[,],[,]]])
#W.field('NoLoS')
#W.record('%s' % (str(


NOLoS_pyshp = shapefile.Writer(shapefile.POLYLINE)
LoS_pyshp = shapefile.Writer(shapefile.POLYLINE)

NOLoS_field_list = []
NOLoS_record_list = []

LoS_field_list = []
LoS_record_list = []


for timestamp in subset_sorted_timestamps:
    df = trace_network_data[timestamp]

    dummy_record = str(timestamp)
    for i in df.index:

        if df.num_buildings[i]>0: 
            NOLoS_lines_list.append([df.Alonglat[0][0],df.Alonglat[0][1]])
            #NOLoS_pyshp.line(parts = [[[df.Alonglat[0][0],df.Alonglat[0][1]],[df.Blonglat[0][0],df.Blonglat[0][1]]]])
            #NOLoS_pyshp.field('NOLoS') #,'crap','asdf')
            #NOLoS_pyshp.record('W') #,'crfd')

            NOLoS_field_list.append(str(timestamp))
            NOLoS_record_list.append(str(timestamp))

        if df.num_buildings[i]<1: 
            LoS_pyshp.line(parts = [[[df.Alonglat[0][0],df.Alonglat[0][1]],[df.Blonglat[0][0],df.Blonglat[0][1]]]])

            LoS_field_list.append(str(timestamp))
            LoS_record_list.append(str(timestamp))

            #LoS_pyshp.field('LoS') #,'morecarap','ffddd')
            #LoS_pyshp.record('Q') #,'crap')
            #LoS_pyshp.field('FIRST_FLD','C','40')
            #LoS_pyshp.field('SECOND_FLD','C','40')
            #LoS_pyshp.record('First','Line')
            #LoS_pyshp.record('Second','Line')


        

NOLoS_pyshp.field('b')
NOLoS_pyshp.record('a')
NOLoS_pyshp.save('NOLoS_SF_data')

LoS_pyshp.field(LoS_field_list)
LoS_pyshp.record(LoS_record_list)
LoS_pyshp.save('LoS_SF_data')

testline = shapefile.Writer(shapefile.POLYLINE)
testline.line(parts = [[[df.Alonglat[0][0],df.Alonglat[0][1]],[df.Blonglat[0][0],df.Blonglat[0][1]]]])
testline.line(parts = [[[df.Alonglat[2][0],df.Alonglat[2][1]],[df.Blonglat[2][0],df.Blonglat[2][1]]]])
testline.field('b')
testline.record('a')
testline.save('SF_test_line')



"""
w = shapefile.Writer(shapefile.POLYLINE)
w.line(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
w.poly(parts=[[[1,3],[5,3]]], shapeType=shapefile.POLYLINE)
w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record('First','Line')
w.record('Second','Line')
w.save('shapefiles/test/line')

infected_list = [298]

# esy now... steady... set()

#for each 'data panel'/pd.dataframe, check if taxi_id in infected_list is able to communicate with another...

#taxi_id_test_list = df.taxiAid.unique().tolist()
#taxi_id_test_list.extend(df.taxiBid.tolist())

taxis_ids_test = df.taxiAid
taxis_ids_test.append(df.taxiBid)


soon_to_be_infected = set(infected_list).intersection(taxis_ids_test.tolist())
 
#might need to think this a tad more carefully.




###################################################################################################################################










# Returns a 3x3 distance matrix for CH:
curl 'http://router.project-osrm.org/table/v1/driving/13.388860,52.517037;13.397634,52.529407;13.428555,52.523219?annotations=distance'



def table(coords_src, coords_dest,
          ids_origin=None, ids_dest=None,
          output='np', minutes=False,
          url_config=RequestConfig, send_as_polyline=True, annotations='distance'):

    Function wrapping OSRM 'table' function in order to get a matrix of
    time distance as a numpy array or as a DataFrame

    Parameters
    ----------

    coords_src : list
        A list of coord as (lat, long) , like :
             list_coords = [(21.3224, 45.2358),
                            (21.3856, 42.0094),
                            (20.9574, 41.5286)] (coords have to be float)
    coords_dest : list, optional
        A list of coord as (lat, long) , like :
             list_coords = [(21.3224, 45.2358),
                            (21.3856, 42.0094),
                            (20.9574, 41.5286)] (coords have to be float)
    ids_origin : list, optional
        A list of name/id to use to label the source axis of
        the result `DataFrame` (default: None).
    ids_dest : list, optional
        A list of name/id to use to label the destination axis of
        the result `DataFrame` (default: None).
    output : str, optional
            The type of durations matrice to return (DataFrame or numpy array)
                'raw' for the (parsed) json response from OSRM
                'pandas', 'df' or 'DataFrame' for a DataFrame
                'numpy', 'array' or 'np' for a numpy array (default is "np")
    url_config: osrm.RequestConfig, optional
        Parameters regarding the host, version and profile to use
    
    annotations : 'distance', returns routing table based on shortest routing distance (metres?)
    taxi_ids = list of taxi identification numbers, usefull only for output distance matrix

    Returns
    -------
        - if output=='raw' : a dict, the parsed json response.
        - if output=='np' : a numpy.ndarray containing the time in minutes,
                            a list of snapped origin coordinates,
                            a list of snapped destination coordinates.
        - if output=='pandas' : a labeled DataFrame containing the time matrix in minutes,
                                a list of snapped origin coordinates,
                                a list of snapped destination coordinates.






"""

















