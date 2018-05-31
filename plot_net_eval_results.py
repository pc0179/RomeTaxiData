# script to load processed taxi trace data, i.e. after network_eval3.py had a bash at it...

import pickle
import numpy as np
import pandas as pd

# get pickle data...
pickle_file = '30S_windows_no_overlap_result_ver1.pickle'

pickle_file = '4_till_11-2-2k14_30s_win_no_overlap_BIG_result.pickle'

pickle_file = 'massive_combined_0_.pickle'

trace_network_data = pd.read_pickle(pickle_file)

timestamps = list(trace_network_data.keys())


# crazy two hop test?

import networkx as nx
G = nx.Graph()
G.add_edges_from([('v1','v2'),('v2','v4'),('v1','v3')])

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)

    return [node for node, length in path_lengths.items() if length == n]

print(neighborhood(G, 'v1', 2))


test_data = trace_network_data[timestamps[0]]


test_data = trace_network_data[timestamps[19910]]
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


import matplotlib.pyplot as plt

plt.plot(timestamps,'o')
plt.show()
