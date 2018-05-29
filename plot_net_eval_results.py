# script to load processed taxi trace data, i.e. after network_eval3.py had a bash at it...

import pickle
import numpy as np
import pandas as pd

# get pickle data...
pickle_file = '30S_windows_no_overlap_result_ver1.pickle'
trace_network_data = pd.read_pickle(pickle_file)

timestamps = list(trace_network_data.keys())


# crazy two hop test?

import networkx as nx
G = nx.Graph()
G.add_edges_from([('v1','v2'),('v2','v4'),('v1','v3')])

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)

    return [node for node, length in path_lengths.items()
                    if length == n]

print(neighborhood(G, 'v1', 1))


test_data = trace_network_data[timestamps[0]]

#def TwoHopLife(df):
df = test_data
edge_pair_list = []

for i in range(len(df)):
    edge_pair_list.append(tuple([str(df.taxiAid[i]),str(df.taxiBid[i])]))

G = nx.Graph()
G.add_edges_from(edge_pair_list)

node = '361'

path_lengths = nx.single_source_dijkstra_path_length(G,node)

n = 2
func_result = [node for node, length in path_lengths.items() if length == n]


