# script to load processed taxi trace data, i.e. after network_eval3.py had a bash at it...

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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

print(neighborhood(G, 'v1', 2))


test_data = trace_network_data[timestamps[0]]


test_data = trace_network_data[timestamps[1100]]

n = 2

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


NoLoSHopCount(test_data,2)

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

ax, ay = zip(*test_data.Alonglat.tolist())
bx, by = zip(*test_data.Blonglat.tolist())

def PlotVANET(df):

    ax, ay = zip(*df.Alonglat.tolist())
    bx, by = zip(*df.Blonglat.tolist())
    num_buildings = test_data.num_buildings.tolist()

    for i in range(len(by)):

        if num_buildings[i]>0:
            plt.plot([ax[i], bx[i]],[ay[i],by[i]],'ro-')

        else:
            plt.plot([ax[i], bx[i]],[ay[i],by[i]],'gd-')


    plt.show()


PlotVANET(test_data)


def CentreOfMassOfTaxis(df,buildings):
""" func. to find Centre of Mass of taxis. For LoS only, set buildings=None, for NoLos set buildings to however many you want included...
"""

    if buildings is None:
        df = df[df.num_buildings<1]
    else:
        df = df[df.num_buildings>buildings]

    ax, ay = zip(*df.Alonglat.tolist())
    bx, by = zip(*df.Blonglat.tolist())

    Xcofm = (np.sum(ax) + np.sum(bx))/(len(ax)+len(bx))
    Ycofm = (np.sum(ay) + np.sum(by))/(len(ay)+len(by))

    return Xcofm, Ycofm

a, b = CentreOfMassOfTaxis(test_data,buildings=None)



#http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#from sklearn.cluster import DBSCAN

#X = test_data.
#db = DBSCAN(eps=0.3, min_samples=10).fit(X)
















