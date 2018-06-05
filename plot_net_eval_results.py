# script to load processed taxi trace data, i.e. after network_eval3.py had a bash at it...

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import osrm

# get pickle data...
#pickle_file = '30S_windows_no_overlap_result_ver1.pickle'

#pickle_file = '4_till_11-2-2k14_30s_win_no_overlap_BIG_result.pickle'

#pickle_file = 'massive_combined_0_.pickle'

pickle_file = '21-2-2k14_0700_1900_taxi_result.pickle'


trace_network_data = pd.read_pickle(pickle_file)

timestamps = list(trace_network_data.keys())


# crazy two hop test?

import networkx as nx
G = nx.Graph()
#G.add_edges_from([('v1','v2'),('v2','v4'),('v1','v3')])
G.add_edges_from([('v1','v2'),('v2','v3'),('v1','v3')])

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


test_data = trace_network_data[timestamps[999]]
CofM_longitude, CofM_latitude = CentreOfMassOfTaxis(test_data,buildings=None)



#http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#from sklearn.cluster import DBSCAN

#X = test_data.
#db = DBSCAN(eps=0.3, min_samples=10).fit(X)



# back to fucking routing/table shit.
def RouteDistanceColumn(df):
    df = test_data
    i = 0
    Rdist_list = []
    for i in range(len(df)):

        table_result = osrm.table([df.Alonglat[i],df.Blonglat[i]],[df.Alonglat[i],df.Blonglat[i]],ids_origin=None, ids_dest=None, output='np', minutes=False, send_as_polyline=True, annotations='distance')

        if table_result == 'Bad Request': rdist = None

        else: rdist = int(round(table_result[table_result>0].min()))

        Rdist_list.append(rdist)

    df['Rdist']=Rdist_list
    return df 


RouteDistanceColumn(test_data)


"""

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

















